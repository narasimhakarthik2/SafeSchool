class CameraLocationMapping:
    def __init__(self):
        self.camera_locations = {
            "Cam1": "Library Corridor - Second Floor",
            "Cam2": "East Wing Corridor - First Floor",
            "Cam3": "West Wing Corridor - First Floor",
            "Cam4": "Cafeteria Entrance - Ground Floor",
            "Cam5": "Main Entrance Lobby - Ground Floor",
            "Cam6": "Science Block Entrance - First Floor",
            "Cam7": "Emergency Exit - Ground Floor",
            "Cam8": "Administrative Block - First Floor"
        }

    def get_location_info(self, camera_id: str) -> dict:
        """Get comprehensive location information for a camera"""
        location = self.camera_locations.get(camera_id)
        if location:
            return {
                "location": location,
            }
        return None