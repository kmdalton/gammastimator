

class symops(dict):
    def __init__(self, libFN=None):
        if libFN is None:
            libFN = dirname(realpath(__file__)) + "/symop.lib"
        self._parse(libFN)
    def _parse(self, libFN):
        with open(libFN, 'r') as f:
            for match in re.findall(r"(?<=\n)[0-9].*?(?=\n[0-9])", '\n'+f.read(), re.DOTALL):
                k = int(match.split()[0])
                self[k] = symop(match)


symops = symops()
