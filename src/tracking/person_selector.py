"""Person selection helpers for multi-person scenes."""

from __future__ import annotations

from src.pose.schemas import FramePoseResult, PersonPose


class PersonSelector:
    """Keep monitoring a chosen track ID; fall back to largest box."""

    def __init__(self, target_person_id: int | None = None) -> None:
        self.target_person_id = target_person_id
        self.locked_id: int | None = target_person_id

    def select(self, frame: FramePoseResult) -> PersonPose | None:
        preferred = self.locked_id if self.locked_id is not None else self.target_person_id
        if preferred is not None:
            for person in frame.persons:
                if person.person_id == preferred:
                    return person
            # Do not silently swap to a different tracked person when the current
            # target is missing from the frame. This keeps ID assignment stable and
            # prevents mixing tracks across adjacent frames.
            self.locked_id = None
            return None

        person = frame.select_person(person_id=None)
        if person is None:
            return None
        self.locked_id = person.person_id
        return person
