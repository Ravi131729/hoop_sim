import sympy as sp
from sympy.physics.mechanics import dynamicsymbols

# -----------------------------------------------------------
# Time and time-dependent variables
# -----------------------------------------------------------
t = sp.symbols('t')

x, y, z = dynamicsymbols('x y z')
psi, th, phi, a = dynamicsymbols('psi th phi a')

# -----------------------------------------------------------
# Rotation matrices
# -----------------------------------------------------------
c = sp.cos; s = sp.sin

Rz = sp.Matrix([
    [c(psi), -s(psi), 0],
    [s(psi),  c(psi), 0],
    [0, 0, 1]
])
Ry = sp.Matrix([
    [ c(th), 0, s(th)],
    [ 0,     1, 0],
    [-s(th), 0, c(th)]
])
Rx_phi = sp.Matrix([
    [1, 0, 0],
    [0, c(phi), -s(phi)],
    [0, s(phi),  c(phi)]
])
Rx_a = sp.Matrix([
    [1, 0, 0],
    [0, c(a), -s(a)],
    [0, s(a),  c(a)]
])

R_SH = sp.simplify(Rz * Ry * Rx_phi)
R_HP = sp.simplify(Rx_a)

# -----------------------------------------------------------
# Position vectors
# -----------------------------------------------------------
rH_S = sp.Matrix([x, y, z])       # ^S r_H
rx, ry, rz = sp.symbols('rx ry rz', real=True)
rP_P = sp.Matrix([rx, ry, rz])    # ^P r_P

# ^S r_P
rP_S = rH_S + R_SH * R_HP * rP_P

# -----------------------------------------------------------
# Time derivative: d/dt (^S r_P)
# -----------------------------------------------------------
rP_S_dot = sp.diff(rP_S, t)

# -----------------------------------------------------------
# Output
# -----------------------------------------------------------
print("\n--- {}^S r_P ---")
sp.pretty_print(sp.simplify(rP_S))

print("\n--- Time derivative d/dt(^S r_P) ---")
sp.pretty_print(sp.simplify(rP_S_dot))
