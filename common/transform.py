# torso
line = rs - ls
theta_t = np.arctan2(line[:, 1], line[:, 0])

# displacement vectors
ra = re - rs
rf = rh - re
la = le - ls
lf = lh - le

# variables for storing fist normalization
ra_n = np.empty_like(ra)
rf_n = np.empty_like(rf)
la_n = np.empty_like(la)
lf_n = np.empty_like(lf)

# 3d rotation matrix, spin theta about z axis
for i, theta in enumerate(theta_t):
    theta = -theta
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])

    ra_n[i, :] = np.matmul(rot, ra[i, :])
    rf_n[i, :] = np.matmul(rot, rf[i, :])
    la_n[i, :] = np.matmul(rot, la[i, :])
    lf_n[i, :] = np.matmul(rot, lf[i, :])

# get motor angles for shoulder
# dc is projection onto yz plane
rdc1 = np.arctan2(ra_n[:, 1], -ra_n[:, 2])
ldc1 = np.arctan2(la_n[:, 1], -la_n[:, 2])

# servo is lift from yz plane
rs1 = np.empty_like(rdc1)
ls1 = np.empty_like(ldc1)
for i in range(ra_n.shape[0]):
    rs1[i] = np.arcsin(ra_n[i, 0] / np.linalg.norm(ra_n[i, :]))
    ls1[i] = np.arcsin(-la_n[i, 0] / np.linalg.norm(la_n[i, :]))

rf_n2 = np.empty_like(rf_n)
lf_n2 = np.empty_like(lf_n)
# 3d rotation matricies, spin about all axis
for i in range(rdc1.shape[0]):
    # define right arm rotation matricies
    rot_xr = np.array([[1, 0, 0],
                       [0, np.cos(-rdc1[i]), -np.sin(-rdc1[i])],
                       [0, np.sin(-rdc1[i]), np.cos(-rdc1[i])]])

    rot_yr = np.array([[np.cos(-rs1[i]), 0, -np.sin(-rs1[i])],
                       [0, 1, 0],
                       [np.sin(-rs1[i]), 0, np.cos(-rs1[i])]])

    # define left arm rotation matricies
    rot_xl = np.array([[1, 0, 0],
                       [0, np.cos(-ldc1[i]), -np.sin(-ldc1[i])],
                       [0, np.sin(-ldc1[i]), np.cos(-ldc1[i])]])

    rot_yl = np.array([[np.cos(ls1[i]), 0, -np.sin(ls1[i])],
                       [0, 1, 0],
                       [np.sin(ls1[i]), 0, np.cos(ls1[i])]])

    # rotate right forearm displacement vector
    temp = np.matmul(rot_xr, rf_n[i, :])
    rf_n2[i, :] = np.matmul(rot_yr, temp)

    # rotate left forarm displacement vector
    temp = np.matmul(rot_xl, lf_n[i])
    lf_n2[i] = np.matmul(rot_yl, temp)

# get motor angles for shoulder
# dc is projection onto yz plane
rdc2 = np.arctan2(rf_n2[:, 0], rf_n2[:, 1])
ldc2 = np.arctan2(-lf_n2[:, 0], lf_n2[:, 1])

# servo is lift from yz plane
rs2 = np.empty_like(rdc2)
ls2 = np.empty_like(ldc2)
for i in range(ra_n.shape[0]):
    rs2[i] = np.arccos(-rf_n2[i, 2] / np.linalg.norm(rf_n2[i, :]))
    ls2[i] = np.arccos(-lf_n2[i, 2] / np.linalg.norm(lf_n2[i, :]))

# plotting
plt.figure(figsize=(8, 6))
plt.plot(rdc1 * 360 / (2 * np.pi), 'r')
plt.plot(rs1 * 360 / (2 * np.pi), 'm')
plt.plot(ldc1 * 360 / (2 * np.pi), 'b')
plt.plot(ls1 * 360 / (2 * np.pi), 'c')
plt.plot(theta_t * 360 / (2 * np.pi), 'k')
plt.title('Motor Angles Shoulder/Torso')
plt.legend(['Right DC', 'Right Servo', 'Left DC', 'Left Servo', 'Torso'])
plt.xlabel('Time (1/60s)')
plt.ylabel('Angle (degrees)')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(rdc2 * 360 / (2 * np.pi), 'r')
plt.plot(rs2 * 360 / (2 * np.pi), 'm')
plt.plot(ldc2 * 360 / (2 * np.pi), 'b')
plt.plot(ls2 * 360 / (2 * np.pi), 'c')
plt.title('Motor Angles Elbow')
plt.legend(['Right DC', 'Right Servo', 'Left DC', 'Left Servo'])
plt.xlabel('Time (1/60s)')
plt.ylabel('Angle (degrees)')
plt.show()

# %%

# saving
path = 'run files/'
np.save(path + 'tdc.npy', theta_t)

np.save(path + 'rdc1.npy', rdc1)
np.save(path + 'rs1.npy', rs1)
np.save(path + 'ldc1.npy', ldc1)
np.save(path + 'ls1.npy', ls1)

np.save(path + 'rdc2.npy', rdc2)
np.save(path + 'rs2.npy', rs2)
np.save(path + 'ldc2.npy', ldc2)
np.save(path + 'ls2.npy', ls2)
