# Function to find equation of line passing through given 2 points.
def equation_line_Oxyz(point_1, point_2):
    """
    Args:
        point_1: [x_1, y_1, z_1]
        point_2: [x_2, y_2, z_2]
    Return:
        [[point_1[0], l], [point_1[1], m], [point_1[2], n]]: List of coefficients to display line equations.
    """

    coeff_x = point_2[0] - point_1[0]
    coeff_y = point_2[1] - point_1[1]
    coeff_z = point_2[2] - point_1[2]

    point_x = point_1[0]
    point_y = point_1[1]
    point_z = point_1[2]
    return [[point_x, coeff_x], [point_y, coeff_y], [point_z, coeff_z]]


# Function to find equation of plane passing through given 3 points.
def equation_plane_Oxyz(point_1, point_2, point_3):
    """
    Args:
        point_1: [x_1, y_1, z_1]
        point_2: [x_2, y_2, z_2]
        point_3: [x_3, y_3, z_3]
    Return:
        [a, b, c, d]: List of coefficients to display plane equations.
    """

    a1 = point_2[0] - point_1[0]
    b1 = point_2[1] - point_1[1]
    c1 = point_2[2] - point_1[2]
    a2 = point_3[0] - point_1[0]
    b2 = point_3[1] - point_1[1]
    c2 = point_3[2] - point_1[2]

    coeff_x = b1 * c2 - b2 * c1
    coeff_y = a2 * c1 - a1 * c2
    coeff_z = a1 * b2 - b1 * a2
    coeff_const = (- coeff_x * point_1[0] - coeff_y * point_1[1] - coeff_z * point_1[2])

    return [coeff_x, coeff_y, coeff_z, coeff_const]


# Function to get the intersection between a line and a plane in 3D space.
def intersection_line_plane_Oxyz(line_coeff, plane_coeff):
    """
    Args:
        line_coeff: List of coefficients to display line equations.
        plane_coeff: List of coefficients to display plane equations.
    Return:
        result_point: Coordinates (x, y, z) found at the intersection.
    """
    line_point_x = line_coeff[0][0]
    line_coeff_x = line_coeff[0][1]
    line_point_y = line_coeff[1][0]
    line_coeff_y = line_coeff[1][1]
    line_point_z = line_coeff[2][0]
    line_coeff_z = line_coeff[2][1]

    plane_coeff_x = plane_coeff[0]
    plane_coeff_y = plane_coeff[1]
    plane_coeff_z = plane_coeff[2]
    plane_coeff_const = plane_coeff[3]

    result_x, result_y, result_z = None, None, None
    denominator = plane_coeff_x*line_coeff_x + plane_coeff_y*line_coeff_y + plane_coeff_z*line_coeff_z
    if denominator:
        t = (-1) * (plane_coeff_const + plane_coeff_x*line_point_x + plane_coeff_y*line_point_y + plane_coeff_z*line_point_z) / denominator
        result_x = line_coeff_x * t + line_point_x
        result_y = line_coeff_y * t + line_point_y
        result_z = line_coeff_z * t + line_point_z
    
    result_point = [result_x, result_y, result_z]
    return result_point
    


if __name__ == "__main__":
    point_1 = [2, 3, 5]
    point_2 = [4, 6, 12]
    line_coeff = equation_line_Oxyz(point_1=point_1, point_2=point_2)
    print("\n\nList of points:")
    print(f"Point 1: {point_1} - Point 2: {point_2}")
    print(
        f"Equation of line is : (x - {line_coeff[0][0]}) / {line_coeff[0][1]} = (y - {line_coeff[1][0]}) / {line_coeff[1][1]} = (z - {line_coeff[2][0]}) / {line_coeff[2][1]}.")

    point_1 = [-1, 2, 1]
    point_2 = [0, -3, 2]
    point_3 = [1, 1, -4]
    plane_coeff = equation_plane_Oxyz(
        point_1=point_1, point_2=point_2, point_3=point_3)
    print("\n\nList of points:")
    print(f"Point 1: {point_1} - Point 2: {point_2} - Point 3: {point_3}")
    print(
        f"Equation of plane is : {plane_coeff[0]}x + {plane_coeff[1]}y + {plane_coeff[2]}z + {plane_coeff[3]} = 0.")

    result_point = intersection_line_plane_Oxyz(line_coeff=line_coeff, plane_coeff=plane_coeff)
    print(f"\n\nCoordinate of the intersection: {result_point}")
