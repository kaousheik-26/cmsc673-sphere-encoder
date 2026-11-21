from manim import *
import json
import numpy as np
from scenes import BG, FN, RN, TRUE, HOW, BAD, MUTE, INK, T, box, arr

config.background_color = BG
CURVE = [json.loads(l) for l in open("/mnt/data1tb/aryan/misc/sphere/runs/framenet_long/tf106m_25k/log.jsonl")]


class Grow106(Scene):
    def construct(self):
        title = T("FrameNet: from 10 M to 106 M parameters", 32).to_edge(UP, buff=0.3)
        self.add(title)
        def stack(n, w, h, color):
            return VGroup(*[RoundedRectangle(corner_radius=0.06, width=w, height=h, stroke_color=color, stroke_width=1.5, fill_color=color, fill_opacity=0.35) for _ in range(n)]).arrange(DOWN, buff=0.06)
        small = stack(4, 1.4, 0.32, RN).move_to(LEFT * 3.2 + DOWN * 0.2)
        big = stack(11, 2.8, 0.32, FN).move_to(LEFT * 3.2 + DOWN * 0.2)
        cap = T("4 blocks, width 384, 6 heads", 22, RN).next_to(small, DOWN, buff=0.25)
        cap2 = T("11 blocks, width 768, 12 heads", 22, FN).next_to(big, DOWN, buff=0.15)
        n = ValueTracker(10.15)
        num = DecimalNumber(10.15, num_decimal_places=1, font_size=64, color=RN)
        num.add_updater(lambda m: m.set_value(n.get_value()).set_color(FN if n.get_value() > 11 else RN))
        unit_t = T("M parameters", 28, INK)
        grp = VGroup(num, unit_t).arrange(RIGHT, buff=0.2).move_to(RIGHT * 2.2 + UP * 1.6)
        num.add_updater(lambda m: m.next_to(unit_t, LEFT, buff=0.2))
        self.play(LaggedStart(*[FadeIn(b, shift=DOWN * 0.1) for b in small], lag_ratio=0.15), FadeIn(cap), FadeIn(grp), run_time=1.4)
        self.wait(0.6)
        self.play(FadeOut(small, scale=0.8), FadeOut(cap), run_time=0.5)
        self.play(LaggedStart(*[GrowFromCenter(b) for b in big], lag_ratio=0.12), FadeIn(cap2), n.animate.set_value(105.89), run_time=2.4)
        num.set_color(FN)
        rows = VGroup(
            T("per block: 2.36 M  to  9.45 M", 22, INK),
            T("attention 2.36 M, MLP 4.72 M, time scale/shift 2.36 M", 18, MUTE),
            T("learning rate 1e-4, warm-up 1000, 25k steps", 22, INK),
            T("bf16 mixed precision, 33 min on one GPU", 22, INK),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.22).next_to(grp, DOWN, buff=0.5).align_to(grp, LEFT).shift(LEFT * 0.6)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.2) for r in rows], lag_ratio=0.3), run_time=1.8)
        self.wait(0.8)
        res = VGroup(T("training clips: 6.0°  to  1.7°", 26, TRUE), T("new clips: 21.3°  to  21.9°", 26, BAD)).arrange(DOWN, aligned_edge=LEFT, buff=0.2).next_to(rows, DOWN, buff=0.5).align_to(rows, LEFT)
        self.play(FadeIn(res[0]))
        self.play(FadeIn(res[1]))
        self.play(Indicate(res[1], color=BAD, scale_factor=1.08))
        self.wait(2.0)


class Objective(Scene):
    def construct(self):
        title = T("The only objective: 1 minus the cosine", 32).to_edge(UP, buff=0.3)
        self.add(title)
        O = LEFT * 4.6 + DOWN * 1.9
        Rr = 3.4
        arc = Arc(radius=Rr, start_angle=0, angle=PI / 2, arc_center=O, color="#6a5a8a", stroke_width=3)
        lab = T("a slice of the sphere", 18, MUTE).next_to(arc, LEFT, buff=0.1).shift(UP * 1.2 + RIGHT * 0.9)
        base = 50.0
        tp = O + Rr * np.array([np.cos(np.deg2rad(base)), np.sin(np.deg2rad(base)), 0])
        truth = Dot(tp, color=TRUE, radius=0.1)
        tl = MathTex("z_t", color=TRUE, font_size=34).next_to(truth, UR, buff=0.05)
        a = ValueTracker(35.0)
        G = lambda: O + Rr * np.array([np.cos(np.deg2rad(base - a.get_value())), np.sin(np.deg2rad(base - a.get_value())), 0])
        guess = always_redraw(lambda: Dot(G(), color=FN, radius=0.1))
        gl = always_redraw(lambda: MathTex(r"\hat z_t", color=FN, font_size=34).next_to(G(), RIGHT, buff=0.08))
        r1 = DashedLine(O, tp, color=MUTE, stroke_width=2)
        r2 = always_redraw(lambda: DashedLine(O, G(), color=MUTE, stroke_width=2))
        wedge = always_redraw(lambda: Angle(Line(O, G()), Line(O, tp), radius=0.9, color=BAD, stroke_width=4) if a.get_value() > 0.8 else VectorizedPoint(O))
        al = always_redraw(lambda: MathTex(r"\alpha", color=BAD, font_size=32).move_to(O + 1.25 * np.array([np.cos(np.deg2rad(base - a.get_value() / 2)), np.sin(np.deg2rad(base - a.get_value() / 2)), 0])))
        self.play(Create(arc), FadeIn(lab), FadeIn(truth), FadeIn(tl), Create(r1))
        self.play(FadeIn(guess), FadeIn(gl), Create(r2))
        self.add(wedge, al)
        ax = Axes(x_range=[0, 40, 10], y_range=[0, 0.25, 0.05], x_length=5.4, y_length=3.6, axis_config={"color": MUTE, "include_numbers": True, "font_size": 20}, tips=False).move_to(RIGHT * 3.2 + DOWN * 0.4)
        xl = T("angle off, degrees", 18, MUTE).next_to(ax.x_axis, DOWN, buff=0.35)
        yl = MathTex(r"\mathcal{L}=1-\cos\alpha", font_size=30, color=BAD).next_to(ax, UP, buff=0.15)
        curve = ax.plot(lambda d: 1 - np.cos(np.deg2rad(d)), x_range=[0, 40], color=BAD, stroke_width=4)
        self.play(Create(ax), FadeIn(xl), FadeIn(yl), Create(curve), run_time=1.4)
        dot = always_redraw(lambda: Dot(ax.c2p(a.get_value(), 1 - np.cos(np.deg2rad(a.get_value()))), color=FN, radius=0.09))
        val = DecimalNumber(0, num_decimal_places=4, font_size=30, color=FN)
        val.add_updater(lambda m: m.set_value(1 - np.cos(np.deg2rad(a.get_value()))))
        vlab = VGroup(T("loss =", 24, FN), val).arrange(RIGHT, buff=0.15).next_to(ax, UP, buff=0.65)
        val.add_updater(lambda m: m.next_to(vlab[0], RIGHT, buff=0.15))
        self.play(FadeIn(dot), FadeIn(vlab))
        self.play(a.animate.set_value(20), run_time=1.6)
        self.wait(0.4)
        self.play(a.animate.set_value(10), run_time=1.2)
        self.wait(0.4)
        self.play(a.animate.set_value(1), run_time=1.4)
        note = VGroup(T("Small misses cost almost nothing: about angle squared / 2", 22, INK),
                      T("1° = 0.00015,  10° = 0.015,  20° = 0.060", 22, MUTE)).arrange(DOWN, buff=0.12).to_edge(DOWN, buff=0.15)
        self.play(FadeIn(note))
        self.wait(1.0)
        n2 = T("One number per guess: it can't say 'too far' or 'wrong way'", 22, BAD).move_to(note[0])
        self.play(FadeOut(note[0]), FadeIn(n2), a.animate.set_value(30), run_time=1.6)
        self.wait(2.0)


class Curve106(Scene):
    def construct(self):
        title = T("106 M while training: memorising, not generalising", 30).to_edge(UP, buff=0.3)
        self.add(title)
        ax = Axes(x_range=[0, 25000, 5000], y_range=[0, 25, 5], x_length=10.5, y_length=5.2, axis_config={"color": MUTE, "font_size": 20}, tips=False).shift(DOWN * 0.35 + LEFT * 0.3)
        xt = VGroup(*[T(f"{k}k" if k else "0", 18, MUTE).next_to(ax.c2p(k * 1000, 0), DOWN, buff=0.15) for k in range(0, 26, 5)])
        yt = VGroup(*[T(f"{v}°", 18, MUTE).next_to(ax.c2p(0, v), LEFT, buff=0.15) for v in range(0, 26, 5)])
        xl = T("training step", 18, MUTE).next_to(ax.x_axis, DOWN, buff=0.55)
        self.play(Create(ax), FadeIn(xt), FadeIn(yt), FadeIn(xl), run_time=1.0)
        frz = DashedLine(ax.c2p(0, 13.62), ax.c2p(25000, 13.62), color=MUTE, stroke_width=2)
        fl = T("freeze first frame, new clips: 13.6°", 18, MUTE).next_to(frz, UP, buff=0.08).align_to(frz, RIGHT)
        self.play(Create(frz), FadeIn(fl))
        pts_tr = [ax.c2p(0, 15.88)] + [ax.c2p(r["step"], r["train_err"]) for r in CURVE]
        pts_te = [ax.c2p(0, 13.62)] + [ax.c2p(r["step"], r["test_err"]) for r in CURVE]
        tr = VMobject(color=FN, stroke_width=4, stroke_opacity=0.55).set_points_as_corners(pts_tr)
        te = VMobject(color=FN, stroke_width=5).set_points_as_corners(pts_te)
        ltr = T("training clips", 20, FN).set_opacity(0.7)
        lte = T("new clips", 20, FN)
        self.play(Create(tr), Create(te), run_time=5, rate_func=linear)
        ltr.next_to(tr.get_end(), UP, buff=0.12).shift(LEFT * 0.6)
        lte.next_to(te.get_end(), DOWN, buff=0.15).shift(LEFT * 0.4)
        self.play(FadeIn(ltr), FadeIn(lte))
        b = min(CURVE, key=lambda r: r["test_err"])
        bd = Dot(ax.c2p(b["step"], b["test_err"]), color=TRUE, radius=0.1)
        bl = T(f"best on new clips: {b['test_err']:.1f}° at step {b['step']}", 20, TRUE).next_to(bd, DR, buff=0.1)
        self.play(FadeIn(bd, scale=1.6), FadeIn(bl))
        last = CURVE[-1]
        ul = T(f"then new clips get worse: {last['test_err']:.1f}°", 20, BAD).move_to(ax.c2p(13000, 23.8))
        self.play(FadeIn(ul), Indicate(te, color=BAD, scale_factor=1.0))
        dl = T(f"while training clips keep improving: {last['train_err']:.1f}°", 20, FN).move_to(ax.c2p(16000, 4.3))
        self.play(FadeIn(dl))
        self.wait(2.5)
