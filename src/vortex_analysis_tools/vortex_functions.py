import numpy as np
from scipy.ndimage import gaussian_filter
import pyvista as pv

def read_solution(fname):
    with open(fname, 'rb') as fid:
        # Read ng (integer)
        ng = np.fromfile(fid, dtype='<i4', count=1)[0]
        print(ng)
        # Read dimensions (3 integers)
        dims = np.fromfile(fid, dtype='<i4', count=3)
        print(np.size(dims))
        print(dims)
        # Read scalars (single-precision floats)
        mach = np.fromfile(fid, dtype='<f4', count=1)[0]
        aoa = np.fromfile(fid, dtype='<f4', count=1)[0]
        reyn = np.fromfile(fid, dtype='<f4', count=1)[0]
        tau = np.fromfile(fid, dtype='<f4', count=1)[0]
        
        print(mach)
        print(aoa)
        print(reyn)
        print(tau)

        # Calculate data length (removed ng)
        datalength = np.prod(dims) * 5
        print(datalength)

        # Read the q array (single-precision float)
        q = np.fromfile(fid, dtype='<f4', count=datalength)
       
        # Check data length
        if len(q) != datalength:
            raise ValueError('Data length mismatch')

        # Reshape q
        q = np.reshape(q,(dims[0], dims[1], dims[2], 5), order="F")
        print(np.size(q))

        # Create data array
        data = np.array([mach, aoa, reyn, tau])

    return data, q

#-------------------------------------------------------------------------------------------------------------#
 
def read_grid_file(fname):
    with open(fname, 'rb') as fid:
        # Determine file size
        fid.seek(0, 2)
        fsize = fid.tell()
        fid.seek(0)

        status = 0
        form = typ = ib = None
        dims = []
        while status == 0:
            # Test for 'STREAM'
            fid.seek(0)
            ng = np.fromfile(fid, dtype=np.int32, count=1)[0]
            dims = np.zeros((3, ng), dtype=int)
            for n in range(ng):
                dims[0, n] = np.fromfile(fid, dtype=np.int32, count=1)[0]
                dims[1, n] = np.fromfile(fid, dtype=np.int32, count=1)[0]
                dims[2, n] = np.fromfile(fid, dtype=np.int32, count=1)[0]
            nelms = np.sum(np.prod(dims, axis=0))
            ibytes = 4 * (1 + ng * 3)
            dbytes = 8 * nelms * 3  # Assuming 'double'
            ibbytes = 4 * nelms

            dt1 = ibytes + dbytes
            dt2 = ibytes + dbytes + ibbytes
            dt3 = ibytes + dbytes // 2
            dt4 = ibytes + dbytes // 2 + ibbytes

            if fsize == dt1:
                form = 'STREAM'
                typ = 'double'
                ib = 0
                status = 1
            elif fsize == dt2:
                form = 'STREAM'
                typ = 'double'
                ib = 1
                status = 1
            elif fsize == dt3:
                form = 'STREAM'
                typ = 'single'
                ib = 0
                status = 1
            elif fsize == dt4:
                form = 'STREAM'
                typ = 'single'
                ib = 1
                status = 1
            else:
                status = 0

            if status == 1:
                break

            # Test for 'UNFORMATTED'
            fid.seek(0)
            rec1 = np.fromfile(fid, dtype=np.int32, count=1)[0]
            ngu = np.fromfile(fid, dtype=np.int32, count=1)[0]
            rec1 = np.fromfile(fid, dtype=np.int32, count=1)[0]
            rec2 = np.fromfile(fid, dtype=np.int32, count=1)[0]
            dimsu = np.zeros((3, ngu), dtype=int)
            for n in range(ngu):
                dimsu[0, n] = np.fromfile(fid, dtype=np.int32, count=1)[0]
                dimsu[1, n] = np.fromfile(fid, dtype=np.int32, count=1)[0]
                dimsu[2, n] = np.fromfile(fid, dtype=np.int32, count=1)[0]
            nelms = np.sum(np.prod(dimsu, axis=0))
            nrecs = 4 + ngu * 2
            ibytes = 4 * (nrecs + 1 + ngu * 3)
            dbytes = 8 * nelms * 3  # Assuming 'double'
            ibbytes = 4 * nelms

            dt1 = ibytes + dbytes
            dt2 = ibytes + dbytes + ibbytes
            dt3 = ibytes + dbytes // 2
            dt4 = ibytes + dbytes // 2 + ibbytes

            if fsize == dt1:
                form = 'UNFORMATTED'
                typ = 'double'
                ib = 0
                status = 1
            elif fsize == dt2:
                form = 'UNFORMATTED'
                typ = 'double'
                ib = 1
                status = 1
            elif fsize == dt3:
                form = 'UNFORMATTED'
                typ = 'single'
                ib = 0
                status = 1
            elif fsize == dt4:
                form = 'UNFORMATTED'
                typ = 'single'
                ib = 1
                status = 1
            else:
                status = 0

        # Read data based on the determined format
        fid.seek(0)
        if form == 'UNFORMATTED':
            rec = np.fromfile(fid, dtype=np.int32, count=1)[0]
            ng = np.fromfile(fid, dtype=np.int32, count=1)[0]
            rec = np.fromfile(fid, dtype=np.int32, count=1)[0]
            rec = np.fromfile(fid, dtype=np.int32, count=1)[0]
            dims = np.zeros((3, ng), dtype=int)
            for n in range(ng):
                dims[0, n] = np.fromfile(fid, dtype=np.int32, count=1)[0]
                dims[1, n] = np.fromfile(fid, dtype=np.int32, count=1)[0]
                dims[2, n] = np.fromfile(fid, dtype=np.int32, count=1)[0]
            rec = np.fromfile(fid, dtype=np.int32, count=1)[0]

            imax = dims[0, :].max()
            jmax = dims[1, :].max()
            kmax = dims[2, :].max()

            x = np.zeros((imax, jmax, kmax, ng))
            y = np.zeros((imax, jmax, kmax, ng))
            z = np.zeros((imax, jmax, kmax, ng))
            iblank = np.ones((imax, jmax, kmax, ng))

            for n in range(ng):
                bytes_read = np.fromfile(fid, dtype=np.int32, count=1)[0]

                for k in range(dims[2, n]):
                    for j in range(dims[1, n]):
                        size = dims[0, n]
                        x[:size, j, k, n] = np.fromfile(fid, dtype=typ, count=size)

                for k in range(dims[2, n]):
                    for j in range(dims[1, n]):
                        size = dims[0, n]
                        y[:size, j, k, n] = np.fromfile(fid, dtype=typ, count=size)

                for k in range(dims[2, n]):
                    for j in range(dims[1, n]):
                        size = dims[0, n]
                        z[:size, j, k, n] = np.fromfile(fid, dtype=typ, count=size)

                if ib == 1:
                    for k in range(dims[2, n]):
                        for j in range(dims[1, n]):
                            size = dims[0, n]
                            iblank[:size, j, k, n] = np.fromfile(fid, dtype=np.int32, count=size)

                if fid.tell() < fsize:
                    rec = np.fromfile(fid, dtype=np.int32, count=1)[0]

        elif form == 'STREAM':
            ng = np.fromfile(fid, dtype=np.int32, count=1)[0]
            dims = np.zeros((3, ng), dtype=int)
            for n in range(ng):
                dims[0, n] = np.fromfile(fid, dtype=np.int32, count=1)[0]
                dims[1, n] = np.fromfile(fid, dtype=np.int32, count=1)[0]
                dims[2, n] = np.fromfile(fid, dtype=np.int32, count=1)[0]

            imax = dims[0, :].max()
            jmax = dims[1, :].max()
            kmax = dims[2, :].max()

            x = np.zeros((imax, jmax, kmax, ng))
            y = np.zeros((imax, jmax, kmax, ng))
            z = np.zeros((imax, jmax, kmax, ng))
            iblank = np.ones((imax, jmax, kmax, ng))

            for n in range(ng):
                for k in range(dims[2, n]):
                    for j in range(dims[1, n]):
                        size = dims[0, n]
                        x[:size, j, k, n] = np.fromfile(fid, dtype=typ, count=size)

                for k in range(dims[2, n]):
                    for j in range(dims[1, n]):
                        size = dims[0, n]
                        y[:size, j, k, n] = np.fromfile(fid, dtype=typ, count=size)

                for k in range(dims[2, n]):
                    for j in range(dims[1, n]):
                        size = dims[0, n]
                        z[:size, j, k, n] = np.fromfile(fid, dtype=typ, count=size)

                if ib == 1:
                    for k in range(dims[2, n]):
                        for j in range(dims[1, n]):
                            size = dims[0, n]
                            iblank[:size, j, k, n] = np.fromfile(fid, dtype=np.int32, count=size)
        
        x = np.squeeze(x)
        y = np.squeeze(y)
        z = np.squeeze(z)
        iblank = np.squeeze(iblank)
        return x, y, z, iblank, dims

#-------------------------------------------------------------------------------------------------------------# 

def calculate_q_criterion(u, v, w, dx, dy, dz):
    """
    Calculates the Q-criterion for vortex identification.

    Args:
        u (np.ndarray): 3D array of u-velocity component.
        v (np.ndarray): 3D array of v-velocity component.
        w (np.ndarray): 3D array of w-velocity component.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        dz (float): Grid spacing in the z-direction.

    Returns:
        np.ndarray: 3D array of Q-criterion values.
    """
    du_dx = np.gradient(u, dx, axis=0)
    dv_dy = np.gradient(v, dy, axis=1)
    dw_dz = np.gradient(w, dz, axis=2)

    du_dy = np.gradient(u, dy, axis=1)
    dv_dx = np.gradient(v, dx, axis=0)

    du_dz = np.gradient(u, dz, axis=2)
    dw_dx = np.gradient(w, dx, axis=0)

    dv_dz = np.gradient(v, dz, axis=2)
    dw_dy = np.gradient(w, dy, axis=1)

    Omega_ij = 0.5 * (np.array([
        [0, du_dy - dv_dx, du_dz - dw_dx],
        [dv_dx - du_dy, 0, dv_dz - dw_dy],
        [dw_dx - du_dz, dw_dy - dv_dz, 0]
    ]))

    S_ij = 0.5 * (np.array([
        [2 * du_dx, du_dy + dv_dx, du_dz + dw_dx],
        [dv_dx + du_dy, 2 * dv_dy, dv_dz + dw_dy],
        [dw_dx + du_dz, dw_dy + dv_dz, 2 * dw_dz]
    ]))

    Q = 0.5 * (np.trace(Omega_ij @ Omega_ij, axis1=-2, axis2=-1) - np.trace(S_ij @ S_ij, axis1=-2, axis2=-1))
    return Q

#-------------------------------------------------------------------------------------------------------------#
    
def calculate_lambda2(u, v, w, dx, dy, dz):
    """
    Calculates the Lambda-2 criterion for vortex core identification.

    Args:
        u (np.ndarray): 3D array of u-velocity component.
        v (np.ndarray): 3D array of v-velocity component.
        w (np.ndarray): 3D array of w-velocity component.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        dz (float): Grid spacing in the z-direction.

    Returns:
        np.ndarray: 3D array of Lambda-2 values.
    """
    du_dx = np.gradient(u, dx, axis=0)
    dv_dy = np.gradient(v, dy, axis=1)
    dw_dz = np.gradient(w, dz, axis=2)

    du_dy = np.gradient(u, dy, axis=1)
    dv_dx = np.gradient(v, dx, axis=0)

    du_dz = np.gradient(u, dz, axis=2)
    dw_dx = np.gradient(w, dx, axis=0)

    dv_dz = np.gradient(v, dz, axis=2)
    dw_dy = np.gradient(w, dy, axis=1)

    Sij = 0.5 * (np.array([
        [2 * du_dx, du_dy + dv_dx, du_dz + dw_dx],
        [dv_dx + du_dy, 2 * dv_dy, dv_dz + dw_dy],
        [dw_dx + du_dz, dw_dy + dv_dz, 2 * dw_dz]
    ]))

    Omega_ij = 0.5 * (np.array([
        [0, du_dy - dv_dx, du_dz - dw_dx],
        [dv_dx - du_dy, 0, dv_dz - dw_dy],
        [dw_dx - du_dz, dw_dy - dv_dz, 0]
    ]))

    M = Sij @ Sij + Omega_ij @ Omega_ij
    lambda2 = np.linalg.eigvals(M.transpose(2, 3, 0, 1))[:, :, 1].real  # Extract the second eigenvalue

    return lambda2

#-------------------------------------------------------------------------------------------------------------# 

def extract_vortex_centerline(grid, lambda2_values, threshold=-0.01, smoothing_iterations=5, smoothing_factor=0.1):
    """
    Extracts the approximate centerline of vortices based on the Lambda-2 criterion.

    Args:
        grid (pyvista.StructuredGrid): The PyVista grid.
        lambda2_values (np.ndarray): 3D array of Lambda-2 values.
        threshold (float): Threshold for identifying vortex cores (Lambda-2 < threshold).
        smoothing_iterations (int): Number of smoothing iterations for the centerline.
        smoothing_factor (float): Smoothing factor (0 to 1).

    Returns:
        pyvista.PolyData or None: A PolyData object representing the vortex centerline, or None if no vortex is found.
    """
    point_indices = np.where(lambda2_values < threshold)
    if not point_indices[0].size:
        return None

    vortex_points = np.vstack((grid.points[point_indices[0], 0],
                               grid.points[point_indices[1], 1],
                               grid.points[point_indices[2], 2])).T

    if vortex_points.shape[0] < 2:
        return None

    # Simple sorting by x-coordinate as a rough approximation of centerline
    sort_indices = np.argsort(vortex_points[:, 0])
    centerline_points = vortex_points[sort_indices]

    # Optional smoothing
    for _ in range(smoothing_iterations):
        centerline_points_smoothed = centerline_points.copy()
        for i in range(1, len(centerline_points) - 1):
            centerline_points_smoothed[i] = (
                (1 - smoothing_factor) * centerline_points[i] +
                0.5 * smoothing_factor * (centerline_points[i - 1] + centerline_points[i + 1])
            )
        centerline_points = centerline_points_smoothed

    return pv.PolyData(centerline_points)

#-------------------------------------------------------------------------------------------------------------#
 

