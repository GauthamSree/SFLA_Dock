import numpy as np

from scipy.cluster.vq import kmeans2
from scipy.spatial import distance
from utils import Complex

def points_on_sphere(number_of_points):
    """Creates a list of points using a spiral method.
    Based on method of 'Minimal Discrete Energy on the Sphere' (E. B. Saff, E.A.
    Rakhmanov and Y.M. Zhou), Mathematical Research Letters, Vol. 1 (1994), pp. 647-662.
    Spiral method: Spiral from top of sphere to bottom of sphere, with points
    places at distances the same as the distance between coils of the spiral.
    """
    
    points = []
    increment = np.pi * (3. - np.sqrt(5.))
    offset = 2. / number_of_points
    for point in range(number_of_points):
        y = point * offset - 1.0 + (offset / 2.0)
        r = np.sqrt(1 - y*y)
        phi = point * increment
        points.append([np.cos(phi) * r, y, np.sin(phi) * r])
    return points


def calculate_initial_poses(receptor: Complex, ligand: Complex, num_points: int, divide_by=2.0, num_sphere_points: int=20):
    """Calculates the position of num_points on the surface of the given protein"""

    distances_matrix_rec = distance.pdist(receptor.atom_coord)
    receptor_max_diameter = np.max(distances_matrix_rec)
    distances_matrix_lig = distance.pdist(ligand.atom_coord )
    ligand_max_diameter = np.max(distances_matrix_lig)
    surface_distance = ligand_max_diameter / divide_by

    coords = receptor.coord

    # Surface clusters
    if len(coords) > num_points:
        surface_clusters = kmeans2(data=coords, k=num_points, minit='points', iter=100)
        surface_centroids = surface_clusters[0]
    else:
        surface_centroids = coords

    # Create points over the surface of each surface cluster
    sampling = []
    for sc in surface_centroids:
        sphere_points = np.array(points_on_sphere(num_sphere_points))
        surface_points = sphere_points * surface_distance + 1.5*sc
        sampling.append(surface_points)

    sampling = np.vstack(sampling)

    # Final cluster of points
    if len(sampling) > num_points:
        s_clusters = kmeans2(data=sampling, k=num_points, minit='points', iter=100)
        sampling = s_clusters[0]

    return sampling, receptor_max_diameter, ligand_max_diameter

def generate_new_pose(rec_coord, ligand_max_diameter, rng):
    surface_distance = ligand_max_diameter / 2.0
    trans_coord = (rng.random()) * surface_distance + (1.5 * rec_coord)
    return trans_coord