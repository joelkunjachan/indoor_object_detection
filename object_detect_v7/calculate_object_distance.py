def calculate_distance(actual_area, bounding_box_area, focal_length):
    """
    Calculate the distance to an object based on its actual area, area of bounding box in an image, and focal length.

    Parameters:
        actual_area (float): The actual area of the object in a known unit (e.g., square meters).
        bounding_box_area (float): The area of the bounding box around the object in pixels.
        focal_length (float): The focal length of the camera in pixels.

    Returns:
        float: The distance to the object in the same unit as actual_area.
    """
    distance = (actual_area * focal_length) / bounding_box_area
    return distance

# Example usage for finding the distance of a person:
actual_person_area = 2.5  # Actual area of the person in square meters
bounding_box_area = 1000  # Area of the bounding box around the person in pixels
focal_length = 1000  # Focal length of the camera in pixels

person_distance = calculate_distance(actual_person_area, bounding_box_area, focal_length)
print("Distance to the person:", person_distance, "meters")
