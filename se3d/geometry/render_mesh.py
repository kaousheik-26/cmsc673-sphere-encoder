"""
Textured-mesh renderer (nvdiffrast) for Google Scanned Objects.

Shading is view-INDEPENDENT (albedo x (ambient + fixed world-space directional
lights)), so every view of an object is photometrically consistent; this is the
ground truth for our probes and GS ceiling, not a match to Blender lighting.
Outputs per view: rgb [H,W,3] in [0,1] composited on white, alpha [H,W],
depth [H,W] (OpenCV z, 0 = background).
"""
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image

from se3d.geometry.cameras import opengl_proj

# OBJ (Y-up) -> world (Z-up), what Blender's default OBJ importer does.
OBJ_TO_ZUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)

LIGHT_DIRS = np.array([[0.4, -0.6, 0.7], [-0.7, 0.5, 0.5], [0.2, 0.8, -0.3]])
LIGHT_W = np.array([0.45, 0.25, 0.10])
AMBIENT = 0.45


class TexturedMesh:
    def __init__(self, obj_path, axis=OBJ_TO_ZUP, normalize="syncdreamer", device="cuda"):
        m = trimesh.load(obj_path, process=False, force="mesh")
        v = np.asarray(m.vertices, dtype=np.float64) @ axis.T
        if normalize == "syncdreamer":  # unit max bbox side, bbox center at origin
            lo, hi = v.min(0), v.max(0)
            v = (v - (lo + hi) / 2) / (hi - lo).max()
        self.scale_info = {"axis": axis.tolist(), "normalize": normalize}
        self.v = torch.tensor(v, dtype=torch.float32, device=device)
        self.f = torch.tensor(np.asarray(m.faces), dtype=torch.int32, device=device)
        vis = m.visual
        uv = getattr(vis, "uv", None)
        img = None
        mat = getattr(vis, "material", None)
        if mat is not None:
            img = getattr(mat, "image", None) or getattr(mat, "baseColorTexture", None)
        if img is None and uv is not None:
            import os.path as osp
            base = osp.dirname(obj_path)
            for cand in ["texture.png", "../materials/textures/texture.png", "../materials/textures/texture.jpg"]:
                if osp.exists(osp.join(base, cand)):
                    img = Image.open(osp.join(base, cand))
                    break
        if uv is None or img is None:
            raise ValueError(f"{obj_path}: no UV texture")
        uv = np.asarray(uv, dtype=np.float32).copy()
        uv[:, 1] = 1.0 - uv[:, 1]  # image rows top-down
        self.uv = torch.tensor(uv, device=device)
        tex = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        self.tex = torch.tensor(tex, device=device)[None].contiguous()
        # smooth vertex normals in world frame
        fn = torch.cross(self.v[self.f[:, 1].long()] - self.v[self.f[:, 0].long()],
                         self.v[self.f[:, 2].long()] - self.v[self.f[:, 0].long()], dim=-1)
        vn = torch.zeros_like(self.v).index_add_(0, self.f[:, 0].long(), fn)
        vn.index_add_(0, self.f[:, 1].long(), fn).index_add_(0, self.f[:, 2].long(), fn)
        self.vn = F.normalize(vn, dim=-1)
        self.device = device


class Renderer:
    def __init__(self, device="cuda"):
        import nvdiffrast.torch as dr

        self.dr = dr
        self.ctx = dr.RasterizeCudaContext(device=device)
        self.device = device
        ld = torch.tensor(LIGHT_DIRS, dtype=torch.float32, device=device)
        self.light_dirs = F.normalize(ld, dim=-1)
        self.light_w = torch.tensor(LIGHT_W, dtype=torch.float32, device=device)

    @torch.no_grad()
    def render(self, mesh, K, w2c, height, width, background=1.0, ssaa=2):
        dr = self.dr
        H, W = height * ssaa, width * ssaa
        Ks = np.array(K, dtype=np.float64).copy()
        Ks[:2] *= ssaa
        w2c = torch.tensor(np.asarray(w2c), dtype=torch.float32, device=self.device)
        v_cam = mesh.v @ w2c[:3, :3].T + w2c[:3, 3]
        v_gl = v_cam * torch.tensor([1.0, -1.0, -1.0], device=self.device)
        P = torch.tensor(opengl_proj(Ks, W, H), dtype=torch.float32, device=self.device)
        clip = torch.cat([v_gl, torch.ones_like(v_gl[:, :1])], -1) @ P.T
        rast, _ = dr.rasterize(self.ctx, clip[None], mesh.f, (H, W))
        uv, _ = dr.interpolate(mesh.uv[None], rast, mesh.f)
        albedo = dr.texture(mesh.tex, uv, filter_mode="linear")
        nrm, _ = dr.interpolate(mesh.vn[None], rast, mesh.f)
        nrm = F.normalize(nrm, dim=-1)
        lam = (nrm @ self.light_dirs.T).clamp_min(0) @ self.light_w
        # two-sided: also light back faces seen through holes
        lam = torch.maximum(lam, (-nrm @ self.light_dirs.T).clamp_min(0) @ self.light_w)
        shade = albedo * (AMBIENT + lam)[..., None]
        zc, _ = dr.interpolate(v_cam[None, :, 2:3].contiguous(), rast, mesh.f)
        alpha = (rast[..., 3:4] > 0).float()
        col = torch.cat([shade * alpha, alpha], -1)  # premultiplied, bg = 0
        col = dr.antialias(col.contiguous(), rast, clip[None], mesh.f)
        # nvdiffrast row 0 = bottom (OpenGL); flip to image convention
        col, zc, alpha = col.flip(1), zc.flip(1), alpha.flip(1)
        rgb, a = col[0, ..., :3], col[0, ..., 3:4].clamp(0, 1)
        if ssaa > 1:
            rgb = F.avg_pool2d(rgb.permute(2, 0, 1)[None], ssaa)[0].permute(1, 2, 0)
            a = F.avg_pool2d(a.permute(2, 0, 1)[None], ssaa)[0].permute(1, 2, 0)
            # depth: nearest valid sample (min over covered sub-pixels)
            zz = torch.where(alpha[0, ..., 0] > 0, zc[0, ..., 0], torch.full_like(zc[0, ..., 0], 1e9))
            zz = -F.max_pool2d(-zz[None, None], ssaa)[0, 0]
            depth = torch.where(zz < 1e8, zz, torch.zeros_like(zz))
        else:
            depth = zc[0, ..., 0] * alpha[0, ..., 0]
        # rgb is premultiplied by coverage via antialias of [shade*1, alpha]; composite on bg
        rgb_c = rgb + (1 - a) * background
        return {"rgb": rgb_c.clamp(0, 1), "alpha": a[..., 0], "depth": depth}


def save_view(out_prefix, view):
    rgb = (view["rgb"].cpu().numpy() * 255 + 0.5).astype(np.uint8)
    a = (view["alpha"].cpu().numpy() * 255 + 0.5).astype(np.uint8)
    Image.fromarray(np.concatenate([rgb, a[..., None]], -1)).save(out_prefix + ".png")
    np.save(out_prefix + "_depth.npy", view["depth"].cpu().numpy().astype(np.float32))
