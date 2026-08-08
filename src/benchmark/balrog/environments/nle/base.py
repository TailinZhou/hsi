from nle import nle_language_obsv
from nle.language_wrapper.wrappers import nle_language_wrapper as language_wrapper
from nle.nethack import USEFUL_ACTIONS
from PIL import Image

from benchmark.balrog.environments import Strings

from ..minihack import ACTIONS as MINIHACK_ACTIONS
from .progress import get_progress_system
from .render import tty_render_image
from .render_rgb import rgb_render_image


class NLELanguageWrapper(language_wrapper.NLELanguageWrapper):
    def __init__(self, env, vlm=False, include_ascii_map=False):
        super().__init__(env, use_language_action=True)
        self.nle_language = nle_language_obsv.NLELanguageObsv()
        self.language_action_space = self.create_action_space()
        self.env = env
        self.vlm = vlm
        self.include_ascii_map = include_ascii_map
        self.done = False

        self.progress = get_progress_system(self.env)
        self.max_steps = self.env.unwrapped._max_episode_steps

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.done = done if not self.done else self.done
        self.progress.update(obs["obs"], reward, self.done, info)
        return obs, reward, self.done, info

    def post_reset(self, obsv):
        return self.post_step(obsv)

    def reset(self, **kwargs):
        self.done = False
        self.progress = get_progress_system(self.env)
        obsv = self.env.reset(**kwargs)
        return self.post_reset(obsv)

    def post_step(self, nle_obsv):
        return self.nle_process_obsv(nle_obsv)

    @property
    def default_action(self):
        if "minihack" in self.env.spec.id.lower():
            return "north"
        else:
            return "esc"

    def get_text_action(self, action):
        return NLELanguageWrapper.all_nle_action_map[self.env.actions[action]][0]

    def nle_process_obsv(self, nle_obsv):
        img = Image.fromarray(self.render("tiles")).convert("RGB") if self.vlm else None
        text = self._format_observation(nle_obsv)
        return {"text": text, "image": img, "obs": nle_obsv}

    def _format_observation(self, nle_obsv):
        """Format native NLE fields into long/short term context dict.

        Matches BALROG's hybrid mode:
        - long_term_context: message + language observation + cursor + ASCII map
          (or statistics when include_ascii_map=False)
        - short_term_context: inventory only
        """
        lang = self.nle_obsv_to_language(nle_obsv)

        cursor = nle_obsv.get("tty_cursor", [0, 0])
        cursor_text = lang["text_cursor"] + f"\n(x={cursor[1]}, y={cursor[0]})"

        long_parts = [
            f"message:\n{lang['text_message']}",
            f"language observation:\n{lang['text_glyphs']}",
            f"cursor:\n{cursor_text}",
        ]

        if self.include_ascii_map:
            ascii_map = self._ascii_render(nle_obsv["tty_chars"])
            ascii_map = "\n".join(ascii_map.split("\n")[1:])  # remove first line (status bar)
            long_parts.append(f"map:\n{ascii_map}")
        else:
            long_parts.append(f"statistics:\n{lang['text_blstats']}")

        short_term = f"inventory:\n{lang['text_inventory']}"
        return {
            "long_term_context": "\n\n".join(long_parts),
            "short_term_context": short_term,
        }

    @staticmethod
    def _ascii_render(chars):
        rows, cols = chars.shape
        result = ""
        for i in range(rows):
            for j in range(cols):
                result += chr(chars[i, j])
            result += "\n"
        return result

    def nle_obsv_to_language(self, nle_obsv):
        """Translate NLE observation into the standard 5-field language dict."""
        message = (
            nle_obsv["text_message"]
            if "text_message" in nle_obsv
            else self.nle_language.text_message(nle_obsv["tty_chars"]).decode("latin-1")
        )

        glyphs = nle_obsv["glyphs"]
        blstats = nle_obsv["blstats"]
        tty_cursor = nle_obsv["tty_cursor"]
        inv_strs = nle_obsv["inv_strs"]
        inv_letters = nle_obsv["inv_letters"]

        return {
            "text_glyphs": self.nle_language.text_glyphs(glyphs, blstats).decode("latin-1"),
            "text_message": message,
            "text_blstats": self.nle_language.text_blstats(blstats).decode("latin-1"),
            "text_inventory": self.nle_language.text_inventory(inv_strs, inv_letters).decode("latin-1"),
            "text_cursor": self.nle_language.text_cursor(glyphs, blstats, tty_cursor).decode("latin-1"),
        }

    def render(self, mode="human"):
        if mode == "tiles":
            obs = self.env.unwrapped.last_observation
            glyphs = obs[self.env.unwrapped._observation_keys.index("glyphs")]
            return rgb_render_image(glyphs)
        elif mode == "tty_image":
            obs = self.env.unwrapped.last_observation
            tty_chars = obs[self.env.unwrapped._observation_keys.index("tty_chars")]
            tty_colors = obs[self.env.unwrapped._observation_keys.index("tty_colors")]
            return tty_render_image(tty_chars, tty_colors)
        else:
            return super().render(mode)

    def get_stats(self):
        return self.progress.__dict__

    def create_action_space(self):
        if "minihack" in self.env.spec.id.lower():
            available_actions = {}
            for action in self.env.actions:
                action_key = NLELanguageWrapper.all_nle_action_map[action][0]
                if action_key not in MINIHACK_ACTIONS:
                    continue
                available_actions[action_key] = MINIHACK_ACTIONS[action_key]

            all_actions = [action for action, _ in available_actions.items()]

        else:
            available_actions = [
                action_strs[0]
                for action, action_strs in NLELanguageWrapper.all_nle_action_map.items()
                if action in USEFUL_ACTIONS
            ]
            single_chars = [chr(i) for i in range(ord("a"), ord("z") + 1)] + [
                chr(i) for i in range(ord("A"), ord("Z") + 1)
            ]
            single_digits = [str(i) for i in range(10)]
            double_digits = [f"{i:02d}" for i in range(100)]
            all_actions = available_actions + single_chars + single_digits + double_digits

        return Strings(all_actions)
