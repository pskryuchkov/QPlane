def check_int(e):
    try:
        int(e)
        return True
    except ValueError:
        return False


def check_float(e):
    try:
        float(e)
        return True
    except ValueError:
        return False


def parse_config(fn):
    with open(fn, "r") as f:
        data = f.readlines()

    lines = [x.strip() for x in data]

    params = {}
    for line in lines:
        cs1 = line.find("#") if line.find("#") > -1 else len(line)
        cs2 = line.find("//") if line.find("//") > -1 else len(line)
        filtered = line[:min(cs1, cs2)]

        if len(filtered) > 0 and "=" in filtered:
            name, val = map(lambda x: x.strip(), filtered.split("="))

            if check_int(val):
                val = int(val)
            elif check_float(val):
                val = float(val)
            elif val[0] == '"' and val[-1] == '"':
                val = val[1:-1]

            params[name] = val

    return params


def get_params(fn):
    class Parameters(object):
        pass

    data = parse_config(fn)
    ps = Parameters()

    for key in data.keys():
        setattr(ps, key, data[key])

    return ps

config = get_params("../quad.cfg")
print config.__dict__