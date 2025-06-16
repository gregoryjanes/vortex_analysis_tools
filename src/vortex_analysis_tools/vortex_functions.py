import numpy as np
from numpy.linalg import eig
from scipy.ndimage import gaussian_filter
import pyvista as pv
from scipy.spatial import Delaunay
from scipy.optimize import minimize
from scipy.interpolate import interpn, RectBivariateSpline
from scipy.ndimage import label
from typing import List, Dict, Any, Tuple


def read_solution(fname):
    with open(fname, 'rb') as fid:
        # Read ng (integer)
        ng = np.fromfile(fid, dtype='<i4', count=1)[0]
        # Read dimensions (3 integers)
        dims = np.fromfile(fid, dtype='<i4', count=3)
        
        # Read scalars (single-precision floats)
        mach = np.fromfile(fid, dtype='<f4', count=1)[0]
        aoa = np.fromfile(fid, dtype='<f4', count=1)[0]
        reyn = np.fromfile(fid, dtype='<f4', count=1)[0]
        tau = np.fromfile(fid, dtype='<f4', count=1)[0]
        
        # Calculate data length (removed ng)
        datalength = np.prod(dims) * 5
       

        # Read the q array (single-precision float)
        q = np.fromfile(fid, dtype='<f4', count=datalength)
       
        # Check data length
        if len(q) != datalength:
            raise ValueError('Data length mismatch')

        # Reshape q
        q = np.reshape(q,(dims[0], dims[1], dims[2], 5), order="F")
        

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
            
            ng = dims.shape[1]
            total_elements = sum(dims[0, n] * dims[1, n] * dims[2, n] for n in range(ng))


            nelms_int = int(total_elements)
            ibytes = 4 * (1 + ng * 3)
            dbytes = 8 * nelms_int * 3  # Assuming 'double'
            ibbytes = 4 * nelms_int

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
  
    du_dx, du_dy, du_dz = np.gradient(u, dx, dy, dz, axis=(0, 1, 2))
    dv_dx, dv_dy, dv_dz = np.gradient(v, dx, dy, dz, axis=(0, 1, 2))
    dw_dx, dw_dy, dw_dz = np.gradient(w, dx, dy, dz, axis=(0, 1, 2))

    # --- Step 2: Calculate Q directly from the derivatives ---
    # The original formula Q = 0.5 * (||Ω||² - ||S||²) can be expanded and simplified
    # to the expression below. This avoids creating arrays for S and Ω.
    
    Q = -0.5 * (du_dx**2 + dv_dy**2 + dw_dz**2) - \
        (du_dy * dv_dx + du_dz * dw_dx + dv_dz * dw_dy)
        
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
    grad_u = np.gradient(u, dx, dy, dz, axis=(0, 1, 2))
    grad_v = np.gradient(v, dx, dy, dz, axis=(0, 1, 2))
    grad_w = np.gradient(w, dx, dy, dz, axis=(0, 1, 2))

    J = np.array([
        [grad_u[0], grad_u[1], grad_u[2]],
        [grad_v[0], grad_v[1], grad_v[2]],
        [grad_w[0], grad_w[1], grad_w[2]]
    ])

    S = 0.5 * (J + np.transpose(J, (1, 0, 2, 3, 4)))
    Omega = 0.5 * (J - np.transpose(J, (1, 0, 2, 3, 4)))
    
    # S^2 + Omega^2 is a symmetric tensor, so we can use eigh for faster computation
    M = np.einsum('ij...,jk...->ik...', S, S) + np.einsum('ij...,jk...->ik...', Omega, Omega)
    
    # Move the 3x3 matrices to the last two dimensions for np.linalg.eigh
    M_reshaped = np.transpose(M, (2, 3, 4, 0, 1))
    
    # Compute eigenvalues for all points at once
    eigenvalues = eigh(M_reshaped).real
    
    # Sort eigenvalues in ascending order and take the middle one (lambda_2)
    lambda2 = np.sort(eigenvalues, axis=-1)[:, :, :, 1]
    
    return lambda2

#-------------------------------------------------------------------------------------------------------------# 

 
def calculate_liutex(u, v, w, dx, dy, dz):
    """
    Calculates the Liutex vector field from 3D velocity field.

    Args:
        u (np.ndarray): 3D array of u-velocity component.
        v (np.ndarray): 3D array of v-velocity component.
        w (np.ndarray): 3D array of w-velocity component.
        dx (float): Grid spacing in the x-direction.
        dy (float): Grid spacing in the y-direction.
        dz (float): Grid spacing in the z-direction.

    Returns:
        tuple: A tuple containing:
            - R (np.ndarray): 3D array of Liutex magnitude.
            - r (np.ndarray): 4D array of Liutex vector direction (3 components x 3D).
    """

    # 1. Calculate velocity gradients
    du_dx = np.gradient(u, dx, axis=0)
    du_dy = np.gradient(u, dy, axis=1)
    du_dz = np.gradient(u, dz, axis=2)

    dv_dx = np.gradient(v, dx, axis=0)
    dv_dy = np.gradient(v, dy, axis=1)
    dv_dz = np.gradient(v, dz, axis=2)

    dw_dx = np.gradient(w, dx, axis=0)
    dw_dy = np.gradient(w, dy, axis=1)
    dw_dz = np.gradient(w, dz, axis=2)

    # 2. Construct the velocity gradient tensor
    grad_v = np.array([
        [du_dx, du_dy, du_dz],
        [dv_dx, dv_dy, dv_dz],
        [dw_dx, dw_dy, dw_dz]
    ])

    # 3. Calculate eigenvalues and eigenvectors
    nz, ny, nx = u.shape
    r = np.zeros((3, nz, ny, nx), dtype=np.complex128)  # Shape: (3, nz, ny, nx)
    R = np.zeros_like(u, dtype=float)

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                grad_v_local = grad_v[:, :, k, j, i]
                eigenvalues, eigenvectors = eig(grad_v_local)

                # Find the real eigenvalue and corresponding eigenvector
                real_eigenvalue_index = np.isreal(eigenvalues)
                if np.any(real_eigenvalue_index):
                    real_eigenvalue_index = np.argmax(eigenvalues[real_eigenvalue_index].real)
                    real_eigenvector = eigenvectors[:, real_eigenvalue_index]
                    
                    # Ensure omega . r > 0 (Eq. 2)
                    vorticity = np.array([dw_dy[k, j, i] - dv_dz[k, j, i],
                                          du_dz[k, j, i] - dw_dx[k, j, i],
                                          dv_dx[k, j, i] - du_dy[k, j, i]])
                    if np.dot(vorticity.real, real_eigenvector.real) < 0:
                        real_eigenvector = -real_eigenvector
                    r[:, k, j, i] = real_eigenvector

                    # Calculate Liutex magnitude (Eq. 3)
                    imaginary_eigenvalues = np.imag(eigenvalues)
                    max_imaginary_eigenvalue = 0
                    for ev in imaginary_eigenvalues:
                      if ev > max_imaginary_eigenvalue:
                        max_imaginary_eigenvalue = ev
                    
                    R[k, j, i] = 2 * max_imaginary_eigenvalue  #Simplified Liutex magnitude

    return R, r

# ----------------------------------------------------------------------------------------

def calculate_liutex_magnitude_gradient(R, dx, dy, dz):
    """Calculates the gradient of the Liutex magnitude."""
    # Ensure correct gradient calculation based on grid axes
    # Assuming axis 0=x, 1=y, 2=z for R[nx,ny,nz]
    # If R is [nz,ny,nx], adjust axes accordingly
    # Example assumes R is [nx, ny, nz] for clarity with dx, dy, dz
    # If R is [nz, ny, nx] use np.gradient(R, dz, dy, dx, axis=(2, 1, 0)) -> check docs
    # Let's assume R is indeed [nx, ny, nz] here. Adjust if not.
    dR_dx = np.gradient(R, dx, axis=0)
    dR_dy = np.gradient(R, dy, axis=1)
    dR_dz = np.gradient(R, dz, axis=2)

    # grad_R should have shape (3, nx, ny, nz) to match r
    grad_R = np.stack([dR_dx, dR_dy, dR_dz], axis=0)
    return grad_R

# ----------------------------------------------------------------------------------------


def pressure_centerline_analysis(q_criterion: np.ndarray, 
                                 p: np.ndarray, 
                                 grid_data: Tuple, 
                                 q_threshold: float = 0.00001) -> List[Dict[str, Any]]:
    """
    Identifies vortex structures and calculates their centerlines based on pressure minima.

    This function is optimized to:
    1.  Handle multiple disconnected vortex structures.
    2.  Filter out small, insignificant noise regions.
    3.  Use bicubic spline interpolation to find sub-pixel pressure minima for a smooth centerline.

    Args:
        q_criterion (np.ndarray): 3D array of Q-criterion values.
        p (np.ndarray): 3D array of pressure values.
        grid_data (Tuple): A tuple containing (x, y, z, dx, dy, dz) grid arrays and spacings.
        q_threshold (float): The threshold for the Q-criterion to define a vortex region.
                             This value is critical and usually needs to be small.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary contains the
                              centerline points and corresponding pressure for a found vortex.
    """
    # --- Step 1: Decompose grid data and identify vortex regions ---
    x, y, z, dx, dy, dz = grid_data
    
    # These are 1D coordinate arrays for lookups later
    x_coords_1d = x[:, 0, 0]
    y_coords_1d = y[0, :, 0]
    z_coords_1d = z[0, 0, :]

    # Create a binary mask where the Q-criterion exceeds the threshold
    binary_q = q_criterion > q_threshold

    # Label contiguous regions in the binary mask. structure=np.ones((3,3,3)) allows for
    # full 3D connectivity (including diagonals).
    labeled_field, num_vortices = label(binary_q, structure=np.ones((3, 3, 3)))

    if num_vortices == 0:
        # This is the most common exit point if q_threshold is too high
        print(f"Warning: No connected regions found with Q-threshold = {q_threshold}. Try a smaller value.")
        return []

    # --- Step 2: Process each identified vortex structure ---
    vortex_analysis_results = []
    min_voxels_for_processing = 20  # Filter out small, noisy regions

    for vortex_label in range(1, num_vortices + 1):
        # Get the (row, col, depth) indices for all voxels in the current vortex
        voxel_indices = np.where(labeled_field == vortex_label)

        if len(voxel_indices[0]) < min_voxels_for_processing:
            continue  # Skip this vortex if it's too small

        centerline_points_phys = []
        pressure_on_centerline = []

        # --- Step 3: Analyze the vortex slice-by-slice to find the centerline ---
        # Loop over every z-slice in the domain as requested.
        for k in range(q_criterion.shape[2]):
            # Find all voxels from the current vortex that are in this specific z-slice
            in_slice_mask = (voxel_indices[2] == k)
            row_indices = voxel_indices[0][in_slice_mask]
            col_indices = voxel_indices[1][in_slice_mask]

            # If the current vortex segment is not in this slice, skip to the next slice.
            if row_indices.size == 0:
                continue

            # Get the bounding box of the vortex cross-section in this slice
            r_min, r_max = np.min(row_indices), np.max(row_indices)
            c_min, c_max = np.min(col_indices), np.max(col_indices)
            
            # BUG FIX & LOGIC: Ensure the bounding box is large enough for bicubic spline (k=3 requires 4+ points)
            if (r_max - r_min + 1) < 4 or (c_max - c_min + 1) < 4:
                continue

            # Extract the physical coordinates and pressure data for the bounding box
            y_slice_coords = y_coords_1d[r_min : r_max + 1]
            x_slice_coords = x_coords_1d[c_min : c_max + 1]
            p_slice_data = p[r_min : r_max + 1, c_min : c_max + 1, k]

            # --- Step 4: Find the sub-pixel pressure minimum using interpolation and optimization ---
            # Create a bicubic spline interpolator for the pressure in the slice
            spline = RectBivariateSpline(y_slice_coords, x_slice_coords, p_slice_data, kx=3, ky=3)

            # Objective function for the optimizer (we want to minimize pressure)
            def objective_function(coords):
                return spline.ev(coords[0], coords[1])

            # Use the minimum pressure point on the coarse grid as a smart initial guess
            p_values_in_mask = p[row_indices, col_indices, k]
            min_idx_in_mask = np.argmin(p_values_in_mask)
            initial_y = y_coords_1d[row_indices[min_idx_in_mask]]
            initial_x = x_coords_1d[col_indices[min_idx_in_mask]]

            # Define the search bounds for the optimizer
            bounds = [(y_slice_coords[0], y_slice_coords[-1]), (x_slice_coords[0], x_slice_coords[-1])]

            # Run the optimizer to find the coordinates of the minimum pressure
            result = minimize(
                objective_function,
                x0=[initial_y, initial_x],
                bounds=bounds,
                method='L-BFGS-B'
            )

            # Store the physical coordinates of the found centerline point and the pressure there
            y_phys, x_phys = result.x
            min_pressure = result.fun
            z_phys = z_coords_1d[k]

            centerline_points_phys.append([x_phys, y_phys, z_phys])
            pressure_on_centerline.append(min_pressure)

        # --- Step 5: Collate results for the processed vortex ---
        if not centerline_points_phys:
            continue

        vortex_analysis_results.append({
            'CenterlinePoints': np.array(centerline_points_phys),
            'PressureOnCenterline': np.array(pressure_on_centerline),
            'Method': 'Min Pressure (Bicubic Interpolation)'
        })

    return vortex_analysis_results
 
def find_constant_grid_bounds(x_coords, y_coords, z_coords, tol=1e-4):

    if not x_coords.ndim == 3:
        raise ValueError("Input coordinate arrays must be 3-dimensional")

    def _find_longest_uniform_run_1d(coord_vector, tol):
        """Helper to find the longest uniform run in a 1D coordinate vector."""
        if len(coord_vector) < 3:
            return None, None

        # Step 1: Calculate the first difference (the spacing)
        d1 = np.diff(coord_vector)
        
        # Step 2: Calculate the second difference (the change in spacing)
        d2 = np.diff(d1)
        
        # Step 3: Find where the change in spacing is essentially zero
        is_flat = np.isclose(d2, 0, atol=tol)
        
        # Step 4: Find the longest contiguous run of "flat" points
        if not np.any(is_flat): return None, None
        
        padded = np.concatenate(([False], is_flat, [False]))
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0] - 1
        
        if len(starts) == 0: return None, None
            
        run_lengths = ends - starts + 1
        longest_run_idx = np.argmax(run_lengths)
        
        d2_start, d2_end = starts[longest_run_idx], ends[longest_run_idx]

        # Step 5: Convert bounds back to original coordinate array indices.
        # A run in the second-difference array from `start` to `end`
        # corresponds to a stable region in the original array from `start` to `end + 2`.
        bounds = (d2_start, d2_end + 2)
        
        # Step 6: Calculate the median spacing from within the identified stable region.
        # This is very robust to any single outlier.
        stable_region_d1 = d1[d2_start : d2_end + 1]
        median_spacing = np.median(stable_region_d1)
        
        return bounds, median_spacing

    # --- Process each dimension ---
    # We take a representative 1D slice for each coordinate
    i_bounds, dx = _find_longest_uniform_run_1d(x_coords[:, 0, 0], tol)
    j_bounds, dy = _find_longest_uniform_run_1d(y_coords[0, :, 0], tol)
    
    # For z, we can assume it's uniform if it has a small standard deviation
    dz1 = np.diff(z_coords[0, 0, :])
    if np.std(dz1) < tol:
        dz = np.median(dz1)
    else:
        # Or run the same robust check if z might also be stretched
        z_bounds, dz = _find_longest_uniform_run_1d(z_coords[0, 0, :], tol)

    if i_bounds is None or j_bounds is None:
        print("Warning: Could not find a contiguous uniform region.")
        return None, None, None
        
    spacing = (dx, dy, dz)
    
    return i_bounds, j_bounds, spacing

