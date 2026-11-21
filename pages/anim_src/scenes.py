from manim import *
import numpy as np

BG = "#1b1b1f"
FN = "#5b9bea"
RN = "#a77fe0"
TRUE = "#4fc28a"
HOW = "#f0a050"
BAD = "#ef6b5b"
MUTE = "#9a9aa2"
INK = "#ececec"
config.background_color = BG


def T(s, size=24, color=INK):
    return Text(s, font_size=size, color=color)


def polar(o, r, deg):
    a = np.deg2rad(deg)
    return o + r * np.array([np.cos(a), np.sin(a), 0.0])


class DirectionGap(Scene):
    def construct(self):
        title = T("Right distance, wrong direction", 34).to_edge(UP, buff=0.35)
        self.add(title)
        O = LEFT * 3.3 + DOWN * 0.5
        r = 2.4
        z0 = Dot(O, color=INK, radius=0.09)
        z0l = T("first frame", 20, MUTE).next_to(z0, DL, buff=0.08)
        Tp = O + RIGHT * r
        tdot = Dot(Tp, color=TRUE, radius=0.11)
        tl = T("true frame", 20, TRUE).next_to(tdot, DOWN, buff=0.15)
        freeze = DashedLine(O, Tp, color=MUTE, stroke_width=3)
        fl = T("freeze: stay put, error = true distance", 20, MUTE).next_to(freeze, UP, buff=0.12).shift(RIGHT * 0.2)
        self.play(FadeIn(z0), FadeIn(z0l), FadeIn(tdot), FadeIn(tl))
        self.play(Create(freeze), FadeIn(fl), run_time=1.2)

        ring = DashedVMobject(Circle(radius=r, color=HOW, stroke_width=3).move_to(O), num_dashes=70)
        ringl = T("RotNet's distance: exactly right", 20, HOW).next_to(ring, UP, buff=0.1)
        self.play(Create(ring), FadeIn(ringl), FadeOut(fl), run_time=1.5)

        g = ValueTracker(0.0)
        G = lambda: polar(O, r, g.get_value())
        arrow = always_redraw(lambda: Arrow(O, G(), buff=0, color=RN, stroke_width=6, max_tip_length_to_length_ratio=0.1))
        gdot = always_redraw(lambda: Dot(G(), color=RN, radius=0.11))
        err = always_redraw(lambda: Line(G(), Tp, color=BAD, stroke_width=5))
        wedge = always_redraw(lambda: Angle(Line(O, Tp), Line(O, G()), radius=0.75, color=RN, stroke_width=4) if g.get_value() > 2 else VectorizedPoint(O))

        s = 0.8
        base = DOWN * 2.3
        bx1, bx2 = RIGHT * 2.7, RIGHT * 4.7
        axis = Line(base + RIGHT * 1.9, base + RIGHT * 5.5, color=MUTE)
        fb = Rectangle(width=1.1, height=r * s, fill_color=MUTE, fill_opacity=0.9, stroke_width=0).move_to(bx1 + base + UP * r * s / 2)
        eb = always_redraw(lambda: (lambda h: Rectangle(width=1.1, height=max(h, 0.001), fill_color=BAD, fill_opacity=0.9, stroke_width=0).move_to(bx2 + base + UP * h / 2))(np.linalg.norm(G() - Tp) * s))
        fbl = T("freeze", 20, MUTE).next_to(bx1 + base, DOWN, buff=0.15)
        ebl = T("guess", 20, BAD).next_to(bx2 + base, DOWN, buff=0.15)
        etitle = T("error", 22, INK).move_to(RIGHT * 3.7 + UP * 1.75)
        deg_num = DecimalNumber(0, num_decimal_places=0, font_size=40, color=RN, unit=r"^\circ")
        deg_lab = VGroup(T("which way off:", 26, RN), deg_num).arrange(RIGHT, buff=0.15)
        deg_lab.move_to(RIGHT * 3.7 + UP * 2.5)
        deg_num.add_updater(lambda m: m.set_value(g.get_value()).next_to(deg_lab[0], RIGHT, buff=0.15))
        ratio = DecimalNumber(0, num_decimal_places=2, font_size=30, color=BAD)
        ratio.add_updater(lambda m: m.set_value(np.linalg.norm(G() - Tp) / r))
        ratio_lab = VGroup(T("guess / freeze =", 22, INK), ratio).arrange(RIGHT, buff=0.12).move_to(RIGHT * 3.7 + DOWN * 3.3)
        ratio.add_updater(lambda m: m.next_to(ratio_lab[0], RIGHT, buff=0.12))

        self.play(FadeIn(arrow), FadeIn(gdot), Create(axis), FadeIn(fb), FadeIn(fbl), FadeIn(ebl), FadeIn(etitle), FadeIn(deg_lab), FadeIn(ratio_lab))
        self.add(err, eb, wedge)
        self.wait(0.6)
        self.play(g.animate.set_value(60), run_time=4, rate_func=smooth)
        even = DashedLine(bx1 + base + UP * r * s + LEFT * 0.8, bx2 + base + UP * r * s + RIGHT * 0.8, color=INK, stroke_width=2)
        evl = T("break-even at 60°", 22, INK).next_to(even, UP, buff=0.1)
        self.play(Create(even), FadeIn(evl))
        self.wait(1.2)
        self.play(g.animate.set_value(82), run_time=2.2)
        note = T("New clips: about 82° off, so moving is worse than not moving", 24, BAD).to_edge(DOWN, buff=0.12).shift(LEFT * 1.2)
        self.play(FadeOut(ratio_lab), FadeIn(note))
        self.wait(1.0)
        self.play(g.animate.set_value(26), run_time=2.2)
        note2 = T("Training clips: about 26° off, so the guess beats freeze", 24, TRUE).move_to(note)
        self.play(FadeOut(note), FadeIn(note2))
        self.wait(2.0)


class TwoHeads(Scene):
    def construct(self):
        title = T("Same guess, two ways to grade it", 34).to_edge(UP, buff=0.3)
        self.add(title)
        L = LEFT * 3.4 + DOWN * 0.5
        R = RIGHT * 3.4 + DOWN * 0.5
        div = DashedLine(UP * 2.6, DOWN * 3.6, color=MUTE, stroke_width=1.5)
        hl = T("FrameNet: one arrow", 26, FN).move_to(L + UP * 2.65)
        hr = T("RotNet: how far + which way", 26, RN).move_to(R + UP * 2.65)
        self.play(FadeIn(div), FadeIn(hl), FadeIn(hr))
        rt, at = 1.8, 25.0
        rg, ag = 1.3, 80.0
        for c in (L, R):
            self.add(Dot(c, color=INK, radius=0.08))
        tL, tR = polar(L, rt, at), polar(R, rt, at)
        truth = VGroup(Dot(tL, color=TRUE, radius=0.1), Dot(tR, color=TRUE, radius=0.1))
        tlabs = VGroup(T("true", 18, TRUE).next_to(tL, UR, buff=0.06), T("true", 18, TRUE).next_to(tR, UR, buff=0.06))
        self.play(FadeIn(truth), FadeIn(tlabs))

        gL = polar(L, rg, ag)
        xi = Arrow(L, gL, buff=0, color=FN, stroke_width=6, max_tip_length_to_length_ratio=0.12)
        xil = MathTex(r"\xi", color=FN, font_size=40).next_to(xi.get_center(), LEFT, buff=0.15)
        self.play(GrowArrow(xi), FadeIn(xil), run_time=1.2)
        gLd = Dot(gL, color=FN, radius=0.1)
        self.add(gLd)
        errL = Line(gL, tL, color=BAD, stroke_width=5)
        lossL = MathTex(r"1-\cos(\hat z_t, z_t)", color=BAD, font_size=34).move_to(L + DOWN * 2.45)
        subL = T("one number: too far? wrong way? can't tell", 19, MUTE).next_to(lossL, DOWN, buff=0.12)
        self.play(Create(errL), FadeIn(lossL), FadeIn(subL))
        self.wait(0.5)

        rad = ValueTracker(0.01)
        ring = always_redraw(lambda: DashedVMobject(Circle(radius=rad.get_value(), color=HOW, stroke_width=3).move_to(R), num_dashes=max(8, int(40 * rad.get_value()))))
        self.add(ring)
        thl = MathTex(r"\hat\theta", r"\text{ how far}", color=HOW, font_size=32).move_to(R + LEFT * 2.55 + UP * 0.25)
        self.play(rad.animate.set_value(rg), FadeIn(thl), run_time=1.4)
        ang = ValueTracker(160.0)
        dvec = always_redraw(lambda: Arrow(R, polar(R, 0.85, ang.get_value()), buff=0, color=RN, stroke_width=6, max_tip_length_to_length_ratio=0.25))
        dl = MathTex(r"\hat d", r"\text{ which way (length 1)}", color=RN, font_size=30).move_to(R + UP * 2.05 + RIGHT * 0.6)
        self.add(dvec)
        self.play(FadeIn(dl), ang.animate.set_value(ag), run_time=1.4)
        gR = always_redraw(lambda: Dot(polar(R, rg, ang.get_value()), color=RN, radius=0.1))
        ray = always_redraw(lambda: DashedLine(R, polar(R, rg, ang.get_value()), color=RN, stroke_width=2))
        self.play(FadeIn(gR), FadeIn(ray))
        self.wait(0.4)

        tring = DashedVMobject(Circle(radius=rt, color=TRUE, stroke_width=1.5).move_to(R), num_dashes=60)
        gap_line = Line(polar(R, rg, at), tR, color=HOW, stroke_width=7)
        wedge = always_redraw(lambda: Angle(Line(R, tR), Line(R, polar(R, 1, ang.get_value())), radius=0.55, color=RN, stroke_width=5) if ang.get_value() - at > 2 else VectorizedPoint(R))
        lossR = MathTex(r"\frac{|\hat\theta-\theta^*|}{10}", r"+", r"\big(1-\cos(\hat d, d^*)\big)", font_size=32).move_to(R + DOWN * 2.5)
        lossR[0].set_color(HOW)
        lossR[2].set_color(RN)
        self.play(FadeIn(tring), Create(gap_line), FadeIn(lossR[0]))
        self.add(wedge)
        self.play(FadeIn(lossR[1]), FadeIn(lossR[2]))
        subR = T("two numbers: we know which half is wrong", 19, MUTE).next_to(lossR, DOWN, buff=0.08)
        self.play(FadeIn(subR))
        self.wait(1.0)
        swap = T("Test: hand RotNet the true direction", 22, TRUE).move_to(R + UP * 2.05 + RIGHT * 0.3)
        self.play(FadeOut(dl), FadeIn(swap))
        self.play(ang.animate.set_value(at), run_time=1.8)
        res = VGroup(T("only the distance error is left", 20, TRUE), T("new clips: 20.4° down to 4.5°", 20, TRUE)).arrange(DOWN, buff=0.06).move_to(swap)
        self.play(FadeOut(swap), FadeIn(res))
        self.wait(2.0)


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


class WalkOnSphere(ThreeDScene):
    def construct(self):
        Rs = 2.3
        self.set_camera_orientation(phi=62 * DEGREES, theta=-40 * DEGREES, zoom=1.3)
        title = T("Walking on the sphere (the exp map)", 32)
        title.to_edge(UP, buff=0.3)
        self.add_fixed_in_frame_mobjects(title)
        ball = Sphere(radius=Rs, resolution=(28, 56), fill_opacity=0.18, stroke_width=0.4, stroke_color="#6a5a8a", checkerboard_colors=["#3a3150", "#332b47"])
        self.play(FadeIn(ball), run_time=1.0)
        n0 = unit([0.62, -0.42, 0.66])
        e1 = unit(np.cross([0, 0, 1], n0))
        e2 = np.cross(n0, e1)
        z0 = Rs * n0
        p0 = Dot3D(z0, color=INK, radius=0.07)
        s1 = T("1. start at the first frame's point", 24, INK).to_corner(DL, buff=0.4)
        self.add_fixed_in_frame_mobjects(s1)
        self.play(FadeIn(p0), FadeIn(s1))
        a = 1.25
        plane = Polygon(z0 + a * (e1 + e2), z0 + a * (-e1 + e2), z0 + a * (-e1 - e2), z0 + a * (e1 - e2), stroke_color=FN, stroke_width=1.5, fill_color=FN, fill_opacity=0.18)
        phi = np.deg2rad(20)
        u = np.cos(phi) * e1 + np.sin(phi) * e2
        Lxi = 1.2
        xi = Arrow3D(z0, z0 + Lxi * u, color=FN, thickness=0.02)
        s2 = T("2. the head outputs an arrow lying flat on the sphere", 24, FN).to_corner(DL, buff=0.4)
        self.add_fixed_in_frame_mobjects(s2)
        self.play(FadeOut(s1), FadeIn(plane), Create(xi), FadeIn(s2), run_time=1.6)
        self.begin_ambient_camera_rotation(rate=0.035)
        th = Lxi / Rs
        arc = ParametricFunction(lambda s: Rs * (np.cos(s) * n0 + np.sin(s) * u), t_range=[0, th], color="#9fe3ff", stroke_width=7)
        zh = Rs * (np.cos(th) * n0 + np.sin(th) * u)
        ph = Dot3D(zh, color="#9fe3ff", radius=0.1)
        s3 = T("3. walk along the surface, same direction, same length", 24, FN).to_corner(DL, buff=0.4)
        eq = MathTex(r"\hat z_t=\cos\theta\,z_0+\sqrt{L}\,\sin\theta\,\tfrac{\xi}{\lVert\xi\rVert}", font_size=34, color=INK).to_corner(UR, buff=0.5).shift(DOWN * 0.6)
        self.add_fixed_in_frame_mobjects(s3, eq)
        self.play(FadeOut(s2), FadeIn(s3), FadeIn(eq), Create(arc), run_time=2.2)
        self.play(FadeIn(ph))
        self.wait(0.5)
        tdir = np.cos(np.deg2rad(75)) * e1 + np.sin(np.deg2rad(75)) * e2
        tht = 0.62
        zt = Rs * (np.cos(tht) * n0 + np.sin(tht) * tdir)
        pt = Dot3D(zt, color=TRUE, radius=0.1)
        hm = unit(zh + zt)
        ang = np.arccos(np.clip(np.dot(unit(zh), unit(zt)), -1, 1))
        b = unit(zt - np.dot(zt, unit(zh)) * unit(zh))
        err = ParametricFunction(lambda s: Rs * (np.cos(s) * unit(zh) + np.sin(s) * b), t_range=[0, ang], color=BAD, stroke_width=6)
        s4 = T("4. the error is the angle between the guess and the true frame", 24, BAD).to_corner(DL, buff=0.4)
        self.add_fixed_in_frame_mobjects(s4)
        self.play(FadeOut(s3), FadeIn(pt), Create(err), FadeIn(s4), run_time=1.6)
        self.wait(1.0)
        ring = ParametricFunction(lambda q: Rs * (np.cos(th) * n0 + np.sin(th) * (np.cos(q) * e1 + np.sin(q) * e2)), t_range=[0, TAU], color=HOW, stroke_width=4)
        s5 = T("RotNet: first pick how far (orange ring), then which way on it", 24, HOW).to_corner(DL, buff=0.4)
        self.add_fixed_in_frame_mobjects(s5)
        self.play(FadeOut(s4), Create(ring), FadeIn(s5), run_time=2.0)
        self.wait(2.5)
        self.stop_ambient_camera_rotation()


def box(text, w, h, color, size=18, fill=0.12):
    r = RoundedRectangle(corner_radius=0.12, width=w, height=h, stroke_color=color, stroke_width=2, fill_color=color, fill_opacity=fill)
    t = Paragraph(*text.split("\n"), font_size=size, color=INK, alignment="center", line_spacing=0.6).move_to(r)
    if t.width > w - 0.15:
        t.scale_to_fit_width(w - 0.15)
    return VGroup(r, t)


def arr(a, b, color=MUTE, sw=3):
    return Arrow(a, b, buff=0.06, color=color, stroke_width=sw, max_tip_length_to_length_ratio=0.18, max_stroke_width_to_length_ratio=10)


class Architecture(Scene):
    def construct(self):
        title = T("Shared body, different heads", 32).to_edge(UP, buff=0.25)
        self.add(title)
        img = box("frame 0\n256x256 image", 1.9, 0.8, MUTE)
        img.move_to([-5.9, 2.3, 0])
        enc = box("frozen sphere\nencoder", 1.9, 0.8, MUTE)
        enc.move_to([-5.9, 1.05, 0])
        grid = VGroup(*[Square(0.16, stroke_width=0.6, stroke_color=INK, fill_color=RN, fill_opacity=0.25 + 0.5 * ((i * 7 + j * 3) % 5) / 5) for i in range(8) for j in range(8)]).arrange_in_grid(8, 8, buff=0.02)
        grid.move_to([-5.9, -0.45, 0])
        gl = MathTex(r"z_0:\ 256\times384", font_size=26, color=INK).next_to(grid, DOWN, buff=0.12)
        self.play(FadeIn(img), run_time=0.5)
        self.play(GrowArrow(arr(img.get_bottom(), enc.get_top())), FadeIn(enc), run_time=0.6)
        self.play(GrowArrow(arr(enc.get_bottom(), grid.get_top())), FadeIn(grid), FadeIn(gl), run_time=0.8)

        lin = box("Linear + position", 2.3, 0.5, RN)
        lin.move_to([-2.0, 2.2, 0])
        blocks = VGroup(*[box("adaLN, attention, MLP", 2.3, 0.48, RN, 16) for _ in range(4)]).arrange(DOWN, buff=0.12).next_to(lin, DOWN, buff=0.3)
        bl = T("x4 transformer blocks", 18, RN).next_to(blocks, DOWN, buff=0.1)
        a1 = arr(grid.get_right() + UP * 0.3, lin.get_left())
        self.play(GrowArrow(a1), FadeIn(lin), run_time=0.6)
        self.play(GrowArrow(arr(lin.get_bottom(), blocks.get_top())), LaggedStart(*[FadeIn(b) for b in blocks], lag_ratio=0.2), FadeIn(bl), run_time=1.2)
        tb = box("t = frame/80\nsin/cos + MLP", 1.9, 0.8, HOW)
        tb.move_to([-5.9, -2.6, 0])
        cl = MathTex("c", font_size=30, color=HOW).next_to(tb.get_right(), UR, buff=0.08).shift(RIGHT * 0.25)
        bus_x = blocks.get_left()[0] - 0.35
        bus = VGroup(Line(tb.get_right(), [bus_x, tb.get_right()[1], 0], color=HOW, stroke_width=3), Line([bus_x, tb.get_right()[1], 0], [bus_x, blocks[0].get_center()[1], 0], color=HOW, stroke_width=3))
        ticks = VGroup(*[arr([bus_x, b.get_center()[1], 0], b.get_left(), HOW, 3) for b in blocks])
        sl = T("time sets scale + shift", 16, HOW).next_to(bus[0], DOWN, buff=0.08).shift(RIGHT * 0.6)
        self.play(FadeIn(tb), Create(bus), FadeIn(cl), run_time=0.8)
        self.play(LaggedStart(*[GrowArrow(t) for t in ticks], lag_ratio=0.2), FadeIn(sl), run_time=0.8)
        out = blocks.get_right() + RIGHT * 0.45
        split = Dot(out, color=INK, radius=0.05)
        self.play(GrowArrow(arr(blocks.get_right(), out, INK)), FadeIn(split), run_time=0.4)

        y1 = 1.55
        f1 = box("Linear\nper token", 1.35, 0.7, FN)
        f2 = box("arrow, flat\non sphere", 1.45, 0.7, FN)
        f3 = box("walk on\nsphere", 1.25, 0.7, FN)
        f4 = MathTex(r"\hat z_t", font_size=36, color=FN)
        for m, x in zip((f1, f2, f3, f4), (1.55, 3.2, 4.8, 6.1)):
            m.move_to([x, y1, 0])
        fh = T("FrameNet head", 20, FN).next_to(f2, UP, buff=0.18)
        fl = MathTex(r"\mathcal{L}=1-\cos(\hat z_t,z_t)", font_size=28, color=BAD).next_to(f3, DOWN, buff=0.22).shift(LEFT * 0.5)
        fa = VGroup(arr(split.get_center(), f1.get_left(), FN), arr(f1.get_right(), f2.get_left(), FN), arr(f2.get_right(), f3.get_left(), FN), arr(f3.get_right(), f4.get_left(), FN))
        self.play(FadeIn(fh), *[FadeIn(m) for m in (f1, f2, f3, f4)], *[GrowArrow(x) for x in fa], run_time=1.4)
        self.play(FadeIn(fl), run_time=0.5)

        r1 = box("Linear, then\nlength 1", 1.45, 0.7, RN)
        r1.move_to([1.6, -0.75, 0])
        r1o = MathTex(r"\hat d", font_size=34, color=RN).move_to([3.0, -0.75, 0])
        r2 = box("mean of tokens\n+ c, MLP", 1.45, 0.7, HOW)
        r2.move_to([1.6, -2.0, 0])
        r2o = MathTex(r"\hat\theta", font_size=34, color=HOW).move_to([3.0, -2.0, 0])
        r3 = box("rotate z0 by\ntheta toward d", 1.55, 0.8, RN)
        r3.move_to([4.6, -1.37, 0])
        r4 = MathTex(r"\hat z_t", font_size=36, color=RN).move_to([6.1, -1.37, 0])
        rh = T("RotNet heads", 20, RN).next_to(r1, UP, buff=0.18)
        ra = VGroup(arr(split.get_center(), r1.get_left(), RN), arr(split.get_center(), r2.get_left(), HOW), arr(r1.get_right(), r1o.get_left(), RN), arr(r2.get_right(), r2o.get_left(), HOW),
                    arr(r1o.get_right(), r3.get_left() + UP * 0.2, RN), arr(r2o.get_right(), r3.get_left() + DOWN * 0.2, HOW), arr(r3.get_right(), r4.get_left(), RN))
        self.play(FadeIn(rh), *[FadeIn(m) for m in (r1, r1o, r2, r2o, r3, r4)], *[GrowArrow(x) for x in ra], run_time=1.6)
        rl = MathTex(r"\mathcal{L}=\tfrac{|\hat\theta-\theta^*|}{10}+\big(1-\cos(\hat d,d^*)\big)", font_size=28, color=BAD).move_to([4.3, -2.75, 0])
        self.play(FadeIn(rl), run_time=0.5)
        self.wait(0.4)
        path_fn = VGroup(a1.copy(), *[x.copy() for x in fa]).set_color(YELLOW)
        path_rn = VGroup(*[x.copy() for x in ra]).set_color(YELLOW)
        self.play(LaggedStart(*[ShowPassingFlash(x.set_stroke(width=6), time_width=0.6) for x in path_fn], lag_ratio=0.35), run_time=2.0)
        self.play(LaggedStart(*[ShowPassingFlash(x.set_stroke(width=6), time_width=0.6) for x in path_rn], lag_ratio=0.25), run_time=2.0)
        note = T("Same body and training data; only the head and the loss change", 20, MUTE).to_edge(DOWN, buff=0.12)
        self.play(FadeIn(note))
        self.wait(2.5)
