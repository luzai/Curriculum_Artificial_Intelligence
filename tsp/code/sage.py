# For small  problem:
# dist=Matrix([[0,1,2],[2,0,1],[1,2,0]])
dist = Matrix(dist)
p = MixedIntegerLinearProgram(maximization=False)  # ,solver="PPL"
x = p.new_variable(nonnegative=True, integer=True)
t = p.new_variable(nonnegative=True, integer=True)
n = dist.nrows()
obj_func = 0
for i in range(n):
    for j in range(n):
        obj_func += x[i, j] * dist[i, j] if i != j else 0
p.set_objective(obj_func)
for i in range(n):
    p.add_constraint(sum([x[i, j] for j in range(n) if i != j]) == 1)
for j in range(n):
    p.add_constraint(sum([x[i, j] for i in range(n) if i != j]) == 1)
for i in range(n):
    for j in range(1, n):
        if i == j:
            continue
        p.add_constraint(t[j] >= t[i] + 1 - n * (1 - x[i, j]))
for i in range(n):
    p.add_constraint(t[i] <= n - 1)

# p.show()
p.solve()
