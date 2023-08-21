import modules.scripts as scripts
import gradio as gr

import scripts.prompt_formating as f

from modules.processing import (
    process_images,
    Processed,
    fix_seed,
    StableDiffusionProcessingImg2Img,
)
from modules.shared import opts, cmd_opts, state

def main(
    p: StableDiffusionProcessingImg2Img,
    frame_count,
    seed_count,
    range_start,
    range_end,
    remove_empty_loras,
    remove_empty_attrs,
    loopback,
):
    positive_prompts = f.build_prompts(
        p.prompt.strip(", "),
        frame_count,
        range_start,
        range_end,
        remove_empty_loras,
        remove_empty_attrs,
    )
    negative_prompts = f.build_prompts(
        p.negative_prompt.strip(", "),
        frame_count,
        range_start,
        range_end,
        remove_empty_loras,
        remove_empty_attrs,
    )

    unformated_prompts = (
        p.prompt.strip(", ") + " ### " + p.negative_prompt.strip(", ") + " "
    )

    p.extra_generation_params["\n\nUnformated Prompts"] = unformated_prompts

    fix_seed(p)
    original_seed = p.seed

    state.job_count = len(positive_prompts) * int(seed_count)

    if loopback:
        original_init_images = p.init_images

    imgs = []
    all_prompts = []
    infotexts = []
    for j in range(int(seed_count)):
        p.seed = original_seed + j
        p.subseed = original_seed + j
        if loopback:
            p.init_images = original_init_images
        for i in range(len(positive_prompts)):
            if state.interrupted:
                break

            if i > 0 and loopback:
                p.init_images = proc.images

            p.prompt = positive_prompts[i]
            p.negative_prompt = negative_prompts[i]
            proc = process_images(p)

            if state.interrupted:
                break

            imgs += proc.images
            all_prompts += proc.all_prompts
            infotexts += proc.infotexts

    return Processed(p, imgs, p.seed, "", all_prompts=all_prompts, infotexts=infotexts)


class Script(scripts.Script):
    is_txt2img = False

    # Function to set title
    def title(self):
        return "fprompt"

    def ui(self, is_img2img):
        with gr.Row():
            frame_count = gr.Number(label="Frame Count", value=2)
            seed_count = gr.Number(label="Seed Count", value=1)
        with gr.Row():
            range_start = gr.Number(label="Range Start", value=0)
            range_end = gr.Number(label="Range End", value=1)
        with gr.Row():
            remove_empty_loras = gr.Checkbox(label="Remove Zeroed LORAs", value=True)
            remove_empty_attrs = gr.Checkbox(
                label="Remove Zeroed Attributes", value=False
            )
            loopback = gr.Checkbox(
                label="Loopback Output", value=False, visible=is_img2img
            )
        with gr.Row():
            gr.Markdown(
                value="""
                ## Usage Guide
                "{\<A\>, \<B\>, [\<start\>=0, \<end\>=1, \<cycles\>=0.5]}" => a value ∈ [\<A\>, \<B\>] that:
                 - = \<A\> when the DPoC(Decimal Percentage of Completion) < \<start\>,
                 - smoothly goes from \<A\> to \<B\> and back again \<cycles\> times while DPoC ∈ [\<start\>, \<end\>], and
                 - = the value it ended up on when DPoC = \<end\> when DPoC > \<end\>

                \[1;2;...:\<A\>,\<B\>,\<start\>,\<end\>\] => 
                a smoothly transitioning lerp between ";" seperated token sets with a default value of \<A\> and a peak value of \<B\>
                
                If the contents of a pair of parentheses can become a number when evaluated, resulting number will replace that string.
                This is the very last step, so {} constructs can be inside and will be evaluated properly


                ### Examples
                 - {0,1} smoothly goes from 0 to 1
                 - {0,1,.5} is 0 for the first half of the prompts and smoothly goes from 0 to 1 in the second half
                 - {0,1,0,.5} smoothy goes from 0 to 1 in the first half of the prompts then stays as 1 for the remainder of the prompts
                 - {0,1,.25,.75,1} the value goes \\_/\\\\_ as it is 0 for the first quarter, cycles up to 1 and back down in the range, then stays constant

                 - [A;B;C:0,1.2,0,.5] => (A:{1.2,0,0,.25}), (B:{0,1.2,0,.5,1}), (C:{0,1.2,.25,.5})

                 - (Example: (5 + (3 * 4 - 2))) => (Example: 15)
                ### Disclaimer
                Some LORAs respond very poorly to a value of 0, so there is an option to have them not appear when the value is 0
                The same is available for parentheses enclosed strings, but that is more for length of prompt and has a significant impact on the consistency between frames
                
                ### Img2Img
                Loopback is available in Img2Img mode to significantly improve consistency between consecutive frames. 
                Currently, some manual setup is required. Generate the first frame in Txt2Img mode then use that as the starting image.
                After that each frame will have the previous frame as its starting state.
                For my uses, a denoising strength of 0.6 - 0.7 works best, but that makes adjacent frames quite similar.
                At very low denoising strengths, the image is very likely to become highly distorted.
                """,
                interactive=False,
            )

        return [
            frame_count,
            seed_count,
            range_start,
            range_end,
            remove_empty_loras,
            remove_empty_attrs,
            loopback,
        ]

    # Function to show the script
    def show(self, is_img2img):
        return True

    # Function to run the script
    def run(
        self,
        p,
        frame_count,
        seed_count,
        range_start,
        range_end,
        remove_empty_loras,
        remove_empty_attrs,
        loopback,
    ):
        # Make a process_images Object
        return main(
            p,
            frame_count,
            seed_count,
            range_start,
            range_end,
            remove_empty_loras,
            remove_empty_attrs,
            loopback,
        )
