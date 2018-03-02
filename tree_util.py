
class Branch:
    def __init__(self, l, r, z):
        self.l = l
        self.r = r
        self.z = z
    def postorder(self, a):
        self.l.postorder(a)
        self.r.postorder(a)
        a.append(1)

class Leaf:
    def __init__(self, z):
        self.z = z
    def postorder(self, a):
        a.append(0)