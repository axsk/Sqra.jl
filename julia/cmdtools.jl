using PyCall
cmd = pyimport("cmdtools")

picking(x, n) = _picking(x, n)[1]
function _picking(x, n)
    y, inds, dists = cmd.estimation.picking_algorithm.picking_algorithm(x', n)
    return y', inds.+1, dists'
end
