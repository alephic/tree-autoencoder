
class Branch:
    def __init__(self, l, r):
        self.l = l
        self.r = r
    def preorder(self, a):
        a.append(1)
        self.l.preorder(a)
        self.r.preorder(a)
    def postorder(self, a):
        self.l.postorder(a)
        self.r.postorder(a)
        a.append(1)

class Leaf:
    def preorder(self, a):
        a.append(0)
    def postorder(self, a):
        a.append(0)

LEAF = Leaf()

def tree_from_encoder_record(action_record):
    stack = []
    for action in action_record:
        if action == 0:
            stack.append(LEAF)
        elif action == 1:
            r = stack.pop()
            l = stack.pop()
            stack.append(Branch(l, r))
    return stack[0]

def _from_decoder_record(action_record):
    if action_record[0] == 0:
        return LEAF, action_record[1:]
    elif action_record[0] == 1:
        l, ar2 = _from_decoder_record(action_record[1:])
        r, ar3 = _from_decoder_record(ar2)
        return Branch(l, r), ar3

def tree_from_decoder_record(action_record):
    return _from_decoder_record(action_record)[0]
