import re


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
        return simplify(eval(string[1:-1]))
    except Exception:
        return string


def cycle(val, cycles):
    temp = val * cycles * 2
    return temp % 1 if temp % 2 < 1 else 1 - temp % 1


def simplify(num):
    if abs(num - round(num)) < 0.00001:
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
        raise ValueError("Start and stop values cannot be the same")
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
        return [fstring(character_prompt, range_start, remove_empty_loras, remove_empty_attrs)]
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
