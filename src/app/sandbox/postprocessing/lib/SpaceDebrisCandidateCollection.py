import os.path


class SpaceDebrisCandidateCollection:
    def __init__(self, candidates):
        self.candidates = {}
        self.add(candidates)

    def add(self, candidates):
        for candidate in candidates:
            self.add_candidate(candidate)

    def add_candidate(self, candidate):
        self.candidates[id(candidate)] = candidate

    def drop_candidate(self, candidate):
        del self.candidates[id(candidate)]

    def view_candidates(self, output_dir, beam):
        detection_boxes = []

        for candidate_id, candidate in self.candidates.iteritems():
            detection_boxes.append(candidate.get_detection_box())

        beam.data.get_view().shapes = detection_boxes

        file_path = os.path.join(output_dir, 'detection_profile')

        beam.data.view(file_path, name = 'detections')

    def save_candidates(self, output_dir):
        for i, (candidate_id, candidate) in enumerate(self.candidates.iteritems()):
            beam_id = 'beam_' + str(candidate.beam.id)
            candidate_id = 'candidate_' + str(i)
            file_path = os.path.join(output_dir, beam_id, 'candidates', candidate_id)

            if not os.path.exists(file_path):
                os.makedirs(file_path)

            # generate table
            candidate.save(file_path = os.path.join(file_path, 'orbit_determination_data'), name = candidate_id)
