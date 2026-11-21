import subprocess, tempfile, os
import mujoco
import imageio.v2 as imageio
import imageio_ffmpeg

class Recorder:
    def __init__(self, env, path="rollout.gif", every=40, fps=20,
                 width=640, height=480, azimuth=135, elevation=-25, distance=1.6,
                 out_width=560):
        self.env, self.path, self.every, self.fps, self.out_width = env, path, every, fps, out_width
        self.renderer = mujoco.Renderer(env.model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0.5, 0.0, 0.3]
        self.cam.azimuth, self.cam.elevation, self.cam.distance = azimuth, elevation, distance
        self.frames, self.n = [], 0

    def __call__(self):                 # called after every sim step
        self.n += 1
        if self.n % self.every == 0:
            self.renderer.update_scene(self.env.data, camera=self.cam)
            self.frames.append(self.renderer.render().copy())

    def save(self):
        tmp = os.path.join(tempfile.gettempdir(), "rec_tmp.mp4")
        imageio.mimsave(tmp, self.frames, fps=self.fps, quality=9)   # near-lossless intermediate
        vf = (f"scale={self.out_width}:-1:flags=lanczos,split[a][b];"
              "[a]palettegen=stats_mode=diff[p];"
              "[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                        "-i", tmp, "-vf", vf, "-loop", "0", self.path], check=True)
        os.remove(tmp)
        print("saved", self.path, len(self.frames), "frames,",
              round(os.path.getsize(self.path) / 1e6, 1), "MB")