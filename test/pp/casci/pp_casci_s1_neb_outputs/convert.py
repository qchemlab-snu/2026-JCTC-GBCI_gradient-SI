from ase.io import read

frames = read("pp_casci_S1_neb.traj", index=":")
print(len(frames))
print(frames[-1])
print(frames[-1].get_positions())