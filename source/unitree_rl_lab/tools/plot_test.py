#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_beta_l_raibert_standalone(save_path="beta_l_standalone.png", show=False):
    # ===== parameters =====
    period = 0.5
    threshold = 0.5
    offset = np.array([0.0, 0.5, 0.5, 0.0])  # FL, FR, RL, RR

    dt = 0.01
    total_time = 2.0
    t = np.arange(0.0, total_time, dt)
    N = len(t)

    # command velocity (m/s)
    vcmd_x, vcmd_y = 0.6, 0.0
    # measured velocity (debug mode: equal to cmd)
    vx, vy = vcmd_x, vcmd_y

    # Raibert gains
    kx, ky = 0.03, 0.03

    # nominal ground height
    z_nominal = -0.28

    # hip positions in BODY frame (placeholder like your file)
    hip_B = np.array([
        [ 0.19,  0.10, 0.0],  # FL
        [ 0.19, -0.10, 0.0],  # FR
        [-0.19,  0.10, 0.0],  # RL
        [-0.19, -0.10, 0.0],  # RR
    ], dtype=float)

    # clamp bounds around hip (BODY frame)
    x_min, x_max = -0.25, 0.25
    y_min, y_max = -0.18, 0.18

    Tst = threshold * period
    Tswing = (1.0 - threshold) * period

    # ===== state/cache =====
    prev_c = np.ones(4, dtype=float)  # start in stance
    p_ref_B = hip_B.copy()
    p_ref_B[:, 2] = z_nominal         # landing ref on ground

    # base motion in world (for visualization only)
    base_pos_w = np.zeros(3)

    # logs
    p_ref_W_hist = np.zeros((N, 4, 3), dtype=float)
    c_ref_hist = np.zeros((N, 4), dtype=float)

    for k in range(N):
        global_phase = ((k * dt) % period) / period
        leg_phase = (global_phase + offset) % 1.0
        c_ref = (leg_phase < threshold).astype(float)  # stance=1

        liftoff = (prev_c > 0.5) & (c_ref < 0.5)        # stance->swing

        swing_phase = (leg_phase - threshold) / (1.0 - threshold)
        swing_phase = np.clip(swing_phase, 0.0, 1.0)

        # Raibert landing reference (computed every step, applied only at liftoff)
        dx = (1.0 - swing_phase) * Tswing * vx + 0.5 * Tst * vx + kx * (vx - vcmd_x)
        dy = (1.0 - swing_phase) * Tswing * vy + 0.5 * Tst * vy + ky * (vy - vcmd_y)

        new_p = hip_B.copy()
        new_p[:, 0] = hip_B[:, 0] + dx
        new_p[:, 1] = hip_B[:, 1] + dy
        new_p[:, 2] = z_nominal

        # clamp around hip
        new_p[:, 0] = np.clip(new_p[:, 0], hip_B[:, 0] + x_min, hip_B[:, 0] + x_max)
        new_p[:, 1] = np.clip(new_p[:, 1], hip_B[:, 1] + y_min, hip_B[:, 1] + y_max)

        # update cache only at liftoff
        p_ref_B[liftoff] = new_p[liftoff]

        # body->world (no rotation for standalone)
        p_ref_W = base_pos_w[None, :] + p_ref_B

        p_ref_W_hist[k] = p_ref_W
        c_ref_hist[k] = c_ref
        prev_c = c_ref

        # move base forward (for visualization only)
        base_pos_w[0] += vcmd_x * dt
        base_pos_w[1] += vcmd_y * dt

    # ===== plotting =====
    leg_names = ["FL", "FR", "RL", "RR"]

    fig = plt.figure(figsize=(14, 5))

    ax3d = fig.add_subplot(131, projection="3d")
    for i in range(4):
        ax3d.plot(p_ref_W_hist[:, i, 0], p_ref_W_hist[:, i, 1], p_ref_W_hist[:, i, 2], label=leg_names[i])
    ax3d.set_title("β_L landing reference (WORLD)")
    ax3d.set_xlabel("x (m)")
    ax3d.set_ylabel("y (m)")
    ax3d.set_zlabel("z (m)")
    ax3d.legend()

    ax_xyz = fig.add_subplot(132)
    ax_xyz.plot(t, p_ref_W_hist[:, 0, 0], label="x_ref")
    ax_xyz.plot(t, p_ref_W_hist[:, 0, 1], label="y_ref")
    ax_xyz.plot(t, p_ref_W_hist[:, 0, 2], label="z_ref")
    ax_xyz.set_title("FL: xyz_ref(t)")
    ax_xyz.set_xlabel("t (s)")
    ax_xyz.grid(alpha=0.3)
    ax_xyz.legend()

    ax_c = fig.add_subplot(133)
    ax_c.step(t, c_ref_hist[:, 0], where="post")
    ax_c.set_title("FL: c_ref(t) (1=stance, 0=swing)")
    ax_c.set_xlabel("t (s)")
    ax_c.set_ylim(-0.2, 1.2)
    ax_c.grid(alpha=0.3)

    plt.tight_layout()

    # ===== output =====
    if save_path:
        fig.savefig(save_path, dpi=200)
        print("✅ Saved figure to:", os.path.abspath(save_path))

    if show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    # 默认：保存到当前目录（你运行命令的那个目录）
    # 如果你本机有桌面想弹窗，把 show=True
    plot_beta_l_raibert_standalone(save_path="beta_l_standalone.png", show=False)
