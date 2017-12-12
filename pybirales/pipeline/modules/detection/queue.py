from pybirales import settings
from pybirales.repository.models import BeamCandidate

import logging as log


class BeamCandidatesQueue:
    def __init__(self, n_beams):
        """
        Initialise the N dimensional queue

        :param n_beams:
        :return:
        """

        self.queue = [[] for _ in range(n_beams)]
        self._max_size = 20
        # self.repository = None
        self.beam_id = None

        self.candidates_to_delete = []

    def set_repository(self, repository):
        """
        Set the repository that will be used by the queue

        :param repository:
        :return:
        """
        # self.repository = repository

    def set_candidates(self, candidates):
        """
        Add candidates to the queue

        :param candidates:
        :return:
        """
        self.queue = candidates

    def get_candidates(self, beam_id):
        """
        Return the candidates associated with a beam, in the queue

        :param beam_id:
        :return:
        """
        return self.queue[beam_id]

    def enqueue(self, new_cluster):
        """

        :param new_cluster:
        :type new_cluster: DetectionCluster
        :return:
        """

        self.beam_id = new_cluster.beam_config['beam_id']
        beam_queue = self.queue[self.beam_id]
        for i, old_cluster in enumerate(beam_queue):
            if old_cluster.to_delete:
                continue

            if new_cluster.is_similar_to(old_cluster, threshold=0.2):
                # Mark old cluster as 'to delete'
                old_cluster.delete()

                # Merge new cluster with the old one and add to the queue
                self.enqueue(old_cluster.merge(new_cluster))

                # Exit the loop once we added the new cluster to the queue
                break
        else:
            # If we didn't merge, add the candidate to the queue
            beam_queue.insert(0, new_cluster)

        # Pop the last element if queue is full
        self.dequeue(self.beam_id)

        # Persist beam candidates
        if settings.detection.save_candidates:
            self.save()

    def dequeue(self, beam_id):
        """
        Pop the last element from the queue if queue has reach maximum size

        :param beam_id:
        :return:
        """

        if len(self.queue[beam_id]) > self._max_size:
            log.debug('Popping last candidate from queue')
            self.queue[beam_id].pop()

    def save(self):
        """
        Save the beam candidates to the database

        :return:
        """

        # Get all clusters, across all queues, which have 'not saved'
        candidates_to_save = [candidate for queue in self.queue for candidate in queue if candidate.to_save]

        # Get all clusters, across all queues, which have 'to delete'
        candidates_to_delete = [candidate for queue in self.queue for candidate in queue if candidate.to_delete]

        # Delete old clusters that were merged
        for candidate in candidates_to_delete:
            bc = BeamCandidate(**candidate.to_json())
            bc.delete()

        # Add new clusters
        for candidate in candidates_to_save:
            bc = BeamCandidate(**candidate.to_json())
            bc.save()
            candidate.to_save = False

        log.info('Saved %s candidates, Deleted %s candidates', len(candidates_to_save), len(candidates_to_delete))

        # Garbage collect - remove candidates marked for deletion from queue
        for q, queue in enumerate(self.queue):
            self.queue[q] = [candidate for candidate in queue if not candidate.to_delete]
