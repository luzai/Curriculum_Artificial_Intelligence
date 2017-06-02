p = MixedIntegerLinearProgram(maximization=False)

f = p.new_variable()
r = p.new_variable()

eps = 1 / (2 * Integer(g.order()))
x = g.vertex_iterator().next()

R = lambda x, y: (x, y) if x < y else (y, x)

# returns the variable corresponding to arc u,v
E = lambda u, v: f[R(u, v)]

# All the vertices have degree 2
for v in g:
    p.add_constraint(sum([f[R(u, v)] for u in g.neighbors(v)]),
                     min=2,
                     max=2)
# r is greater than f
for u, v in g.edges(labels=None):
    p.add_constraint(r[(u, v)] + r[(v, u)] - f[R(u, v)], min=0)
# defining the answer when g is not directed
tsp = Graph()
# no cycle which does not contain x
weight = lambda l: l if (l is not None and l) else 1
for v in g:
    if v != x:
        p.add_constraint(sum([r[(u, v)] for u in g.neighbors(v)]), max=1 - eps)
p.set_objective(sum([weight(l) * E(u, v) for u, v, l in g.edges()]))
p.set_binary(f)
