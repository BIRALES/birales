import os.path
import app.sandbox.postprocessing.config.application as config


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
        detection_data = []
        for candidate_id, candidate in self.candidates.iteritems():
            detection_boxes.append(candidate.get_detection_box())
            detection_data.append(candidate.detection_data)

        # beam.data.get_view().set_shapes(detection_boxes)
        beam.data.get_view().set_detections(detection_data)

        beam_id = 'beam_' + str(beam.id)
        file_path = os.path.join(output_dir, beam_id, config.DETECTIONS_BEAM_FILE_NAME)

        beam.data.view(file_path, name = 'detections')

    def save_candidates(self, output_dir):
        for i, (candidate_id, candidate) in enumerate(self.candidates.iteritems()):
            beam_id = 'beam_' + str(candidate.beam.id)
            candidate_id = 'candidate_' + str(i)
            file_path = os.path.join(output_dir, beam_id, 'candidates', candidate_id)

            if not os.path.exists(file_path):
                os.makedirs(file_path)

            # generate table
            candidate.create_table(file_path = os.path.join(file_path, config.OD_FILE_NAME), name = candidate_id)
