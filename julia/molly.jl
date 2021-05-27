using Molly

timestep = 0.00005 # ps
temp = 10 #298 # K
n_steps = 2_000

atoms, specific_inter_lists, general_inters, nb_matrix, coords, box_size = readinputs(
            joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_top_ff.top"),
            joinpath(dirname(pathof(Molly)), "..", "data", "5XER", "gmx_coords.gro"))

s = Simulation(
    simulator=VelocityVerlet(),
    atoms=atoms,
    specific_inter_lists=specific_inter_lists,
    general_inters=general_inters,
    coords=coords,
    velocities=[velocity(a.mass, temp) * 0.01 for a in atoms],
    temperature=temp,
    box_size=box_size,
    neighbour_finder=DistanceNeighbourFinder(nb_matrix, 10),
    thermostat=AndersenThermostat(0.01),
    loggers=Dict("temp" => TemperatureLogger(10),
                    "writer" => StructureWriter(10, "traj_5XER_1ps.pdb")),
    timestep=timestep,
    n_steps=n_steps
)

simulate!(s)

s.loggers["temp"].temperatures
UnicodePlots.lineplot(s.loggers["temp"].temperatures)