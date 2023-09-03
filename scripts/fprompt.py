import modules.scripts as scripts
import gradio as gr

from modules.processing import (
    process_images,
    Processed,
    fix_seed,
    StableDiffusionProcessing,
    StableDiffusionProcessingImg2Img,
)
from modules.shared import opts, cmd_opts, state

import re

def curve(i, a=0):
    return (1-a)*((2-i)*i)**.5 + a*i

def evalInParens(str):
    out = ""
    while str != out:
        out = str
        str = re.sub(r"\([^\(\)]*\)", applyFunctionToGroup(evalIfEvalable), str)
    return out


def applyFunctionToGroup(callable):
    return lambda obj: callable(obj.group() if obj.group() is not None else "")


def evalIfEvalable(string):
    try:
        return simplify(eval(string[1:-1].replace("<", "(").replace(">", ")")))
    except Exception:
        return string


def cycle(val, cycles):
    temp = val * cycles * 2
    return temp % 1 if temp % 2 < 1 else 1 - temp % 1


def simplify(num):
    if abs(num - round(num)) < 0.001:
        num = round(num)
    num_str = str(num)
    idx_99999 = num_str.find("99999")
    idx_00000 = num_str.find("00000")

    if idx_99999 != -1 and (idx_00000 == -1 or idx_99999 < idx_00000):
        # If '99999' was found and it appears before '00000', round up
        idx = idx_99999
        while num_str[idx] == "9":
            idx -= 1
        if num_str[idx] == ".":
            idx -= 1
        return num_str[:idx] + str(int(num_str[idx]) + 1)
    elif idx_00000 != -1:
        # If '00000' was found, simply remove it and everything after it
        return num_str[:idx_00000]

    if len(num_str) > 6:
        return num_str[:6]

    return num_str


def stringToCurrentState(str, partial):
    return listToCurrentState(partial, *map(lambda x: float(x.strip()), str.split(",")))


def listToCurrentState(partial, first, last, start=0, stop=1, cycles=0.5):
    if start == stop:
        return first if partial < start else last
    if start > stop:
        first, last = last, first
        start, stop = stop, start
    if first == last:
        return first
    if partial <= start:
        return first
    if partial >= stop:
        return first + (last - first) * cycle(1, cycles)

    range = stop - start

    partialHere = cycle((partial - start) / range, cycles)

    valueRange = last - first
    currentValue = first + valueRange * partialHere

    return currentValue


def remove_nested_zero_strings(input_string, type_start, type_end):
    out = input_string
    split = input_string.split(":0" + type_end)
    if len(split) > 1:
        out = ""
        for str in split[:-1]:
            out += str
            depth = 1
            for i in reversed(range(len(out))):
                c = out[i]
                if c == type_start:
                    depth -= 1
                    if depth == 0:
                        out = out[:i]
                        break
                if c == type_end:
                    depth += 1

            if depth > 0:
                raise Exception("Missing " + type_start)
        out += split[-1]

    return out


def getTransitionInfo(a=0, b=1, start=0, end=1):
    return {"a": a, "b": b, "start": start, "end": end}


def distribute(str):
    # [1;2;3;4:a,b,start,end]
    parts = str.split(":")
    phases = parts[0].split(";")
    if len(phases) < 2:
        raise Exception(
            f'There must be at least 2 phases in [] requirement failed in "{str}". '
            + "You may want to instead use the {} system for a range of strengths"
        )
    if len(parts) > 1:
        transitionInfo = parts[1]
    else:
        transitionInfo = ""
    transitionInfo = getTransitionInfo(
        *(map(lambda a: float(a), filter(None, transitionInfo.split(","))))
    )

    a = transitionInfo["a"]
    b = transitionInfo["b"]
    start = transitionInfo["start"]
    end = transitionInfo["end"]

    span = (end - start) / (len(phases) - 1)

    out = ""

    for i in range(len(phases)):
        if i == 0:
            out += (
                "("
                + phases[i]
                + ":{"
                + simplify(b)
                + ","
                + simplify(a)
                + ","
                + simplify(start)
                + ","
                + simplify(start + span)
                + "}), "
            )
        elif i == len(phases) - 1:
            out += (
                "("
                + phases[i]
                + ":{"
                + simplify(a)
                + ","
                + simplify(b)
                + ","
                + simplify(end - span)
                + ","
                + simplify(end)
                + "})"
            )
        else:
            out += (
                "("
                + phases[i]
                + ":{"
                + simplify(a)
                + ","
                + simplify(b)
                + ","
                + simplify(start + (i - 1) * span)
                + ","
                + simplify(start + (i + 1) * span)
                + ",1}), "
            )

    return out


def parseBrackets(str):
    out = ""

    parts = str.split("[")
    for part in parts:
        pieces = part.split("]")
        if len(pieces) == 1:
            out += part
            continue
        out += distribute(pieces[0])
        out += pieces[1]

    return out


def parseCurlies(str, partial):
    parts = str.split("{")

    out = ""

    for part in parts:
        pieces = part.split("}")
        if len(pieces) == 1:
            out += part
            continue
        out += simplify(stringToCurrentState(pieces[0], partial))
        out += pieces[1]

    return out


def fstring(str, partial, remove_empty_loras, remove_empty_attrs):
    out = parseCurlies(str, partial)
    out = evalInParens(out)
    out = parseBrackets(out)
    out = parseCurlies(out, partial)
    out = evalInParens(out)

    if remove_empty_loras:
        out = remove_nested_zero_strings(out, "<", ">")
        out = remove_nested_zero_strings(out, "<", ".0>")

    if remove_empty_attrs:
        out = remove_nested_zero_strings(out, "(", ")")
        out = remove_nested_zero_strings(out, "(", ".0)")
        out = re.sub(r"(, *)+", ", ", out)

    return out


def build_prompts(
    character_prompt,
    frame_count,
    range_start,
    range_end,
    remove_empty_loras,
    remove_empty_attrs,
):
    if range_start == range_end or frame_count == 1:
        return [
            fstring(
                character_prompt, range_start, remove_empty_loras, remove_empty_attrs
            )
        ]
    if frame_count < 1:
        raise ValueError("Frame count must be greater than 1")

    prompts = []

    for i in range(int(frame_count)):
        partial = range_start + (i / (frame_count - 1)) * (range_end - range_start)
        prompts.append(
            fstring(character_prompt, partial, remove_empty_loras, remove_empty_attrs)
        )

    if frame_count <= 0 or frame_count >= len(prompts):
        return prompts

    return prompts[: int(frame_count)]


def extractParam(
    originalVal, prompt: str, paramName: str, normalizingFunction=lambda x: x
):
    parts = (" " + prompt).split(f" {paramName}:")
    if len(parts) == 1:
        return originalVal, prompt
    if len(parts) == 2:
        valueString = re.match(r" ?[0-9.]+", parts[1])[0]
        outValue = normalizingFunction(valueString)
        outPrompt = parts[0] + (
            parts[1][len(valueString) :] if parts[1] != valueString else ""
        )
        return outValue, outPrompt.strip()

    raise ValueError("Duplicate parameter definition of '%s' in prompt." % paramName)


def extractAndApplyTextDefinedParams(p: StableDiffusionProcessing, img2img: bool):
    normalize64 = lambda x: round(float(x) / 64) * 64

    p.cfg_scale, p.prompt = extractParam(p.cfg_scale, p.prompt, "cfg", float)
    p.width, p.prompt = extractParam(p.width, p.prompt, "width", normalize64)
    p.width, p.prompt = extractParam(p.width, p.prompt, "w", normalize64)
    p.height, p.prompt = extractParam(p.height, p.prompt, "height", normalize64)
    p.height, p.prompt = extractParam(p.height, p.prompt, "h", normalize64)
    p.steps, p.prompt = extractParam(p.steps, p.prompt, "steps", int)
    p.seed, p.prompt = extractParam(p.seed, p.prompt, "seed", int)
    p.subseed = p.seed

    if img2img:
        p.denoising_strength, p.prompt = extractParam(p.denoising_strength, p.prompt, "ds", float)
        p.denoising_strength, p.prompt = extractParam(p.denoising_strength, p.prompt, "denoising_strength", float)

    return p


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
    img2img = hasattr(p, "init_images")
    positive_prompts = build_prompts(
        p.prompt.strip(", "),
        frame_count,
        range_start,
        range_end,
        remove_empty_loras,
        remove_empty_attrs,
    )
    negative_prompts = build_prompts(
        p.negative_prompt.strip(", "),
        frame_count,
        range_start,
        range_end,
        remove_empty_loras,
        remove_empty_attrs,
    )

    unformated_prompt = p.prompt.strip(", ")
    unformated_negative_prompt = p.negative_prompt.strip(", ")

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

            p = extractAndApplyTextDefinedParams(p, img2img)

            p.extra_generation_params = {
                "🟢 Unformated Prompt": (unformated_prompt),
                "🔴 Unformated Negative Prompt": (unformated_negative_prompt),
                "🟠 DCoP": (
                    (range_start)
                    if len(positive_prompts) < 2
                    else (
                        range_start
                        + (i / (frame_count - 1)) * (range_end - range_start)
                    )
                ),
                "🟡🟢 DCoP Range Start": range_start,
                "🟡🔴 DCoP Range End": range_end,
                "🔵 Frame": i + 1,
                "🟣 Total Frames": len(positive_prompts),
                "Remove Zero Weighted LORAs": "✔️" if remove_empty_loras else "❌",
                "Remove Zero Weighted Attributes": "✔️" if remove_empty_attrs else "❌",
                "Loopback": "✔️" if loopback else "❌",
            }

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
            remove_empty_loras = gr.Checkbox(label="Remove Zero Weighted LORAs", value=True)
            remove_empty_attrs = gr.Checkbox(label="Remove Zero Weighted Attributes", value=False)
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
                This is the very last step, so {} constructs can be inside and will be evaluated properly.
                Since it is done recursively, you must replace internal () with <>. This allows you to use python builtins like round and int.
                
                I also created the function curve(x, a) that takes an x value from 0 to 1 and optionally a flattening percentage (a):
                it actually returns (1-a)\*sin(arccos(x))+a\*x, which in simple terms is a curve that jumps up then grows more slowly.
                The flattening percentage is basically how much it is squished to a linear growth. 0-> quarter circle, 1 -> straight line

                Additionally, several configuration options can be specified in the text and dynamically changed.
                Simply put the name of the configuration option then : then the number or expression.
                Supported configurations: cfg, seed, steps, width(w), height(h), denoising_strength(ds, only works in Img2Img)

                ### Examples
                 - {0,1} smoothly goes from 0 to 1
                 - {0,1,.5} is 0 for the first half of the prompts and smoothly goes from 0 to 1 in the second half
                 - {0,1,0,.5} smoothy goes from 0 to 1 in the first half of the prompts then stays as 1 for the remainder of the prompts
                 - {0,1,.25,.75,1} the value goes \\_/\\\\_ as it is 0 for the first quarter, cycles up to 1 and back down in the range, then stays constant

                 - [A;B;C:0,1.2,0,.5] => (A:{1.2,0,0,.25}), (B:{0,1.2,0,.5,1}), (C:{0,1.2,.25,.5})

                 - (Example: (5 + (3 * 4 - 2))) => (Example: 15)
                 - (Example: (int\<4/3\>) => (Example: 1)
                 - (Example: (curve\<{1,0}\>)) => (Example: \<x\>) where x is a value that follows the edge of the 1st quadrant of the unit circle
                 - (Example: (curve\<{0,1}\>)) => (Example: \<x\>) where x is a value that follows the edge of the 2nd quadrant of the unit circle
                 - (Example: (1-curve\<{0,1}\>)) => (Example: \<x\>) where x is a value that follows the edge of the 3rd quadrant of the unit circle
                 - (Example: (1-curve\<{1,0}\>)) => (Example: \<x\>) where x is a value that follows the edge of the 4th quadrant of the unit circle
                 - (Example: (curve\<{0,1}, 0.8\>)) => (Example: \<x\>) where x is .2*curve\<{0,1}\> + {0,.8}

                 - cfg: {7,9} => is removed from the prompt and sets the cfg to a value that smoothly goes from 7 to 9 over all frames.

                ### Disclaimer
                Some LORAs respond very poorly to a value of 0, so there is a default enabled option to have them not appear when their weight is 0
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
