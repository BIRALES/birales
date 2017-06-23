from pybirales.modules.detection.repository import BeamCandidateRepository
from pybirales.base import settings
import logging as log
from multiprocessing.queues import Queue


class BeamCandidatesQueue:
    def __init__(self, n_beams):
        self.queue = [[] for _ in range(n_beams)]
        self._max_size = 20
        self.repository = None

        # super(BeamCandidatesQueue, self).__init__(maxsize=self._max_size)
        # self.repository = BeamCandidateRepository()

    def set_repository(self, repository):
        self.repository = repository

    def enqueue(self, new_cluster):
        # Check if beam candidate is similar to candidate that was already added to the queue
        beam_queue = self.queue[new_cluster.beam_id]
        for old_cluster in beam_queue:

            if new_cluster.is_similar_to(old_cluster, threshold=0.2):
                log.debug('Merging clusters in beam_queue %s (length: %s)', new_cluster.beam_id, len(beam_queue))
                # mark old cluster as 'to delete'
                old_cluster.delete()

                # merge new cluster with the old one and add to the queue
                beam_queue.insert(0, old_cluster.merge(new_cluster))

                # Exit the loop once we added the new cluster to the queue
                break
        else:
            # If we didn't merge, add the candidate to the queue
            beam_queue.insert(0, new_cluster)
        # log.debug('Enqueuing took %1.3f s', time.time() - t1)
        # Pop the last element if queue is full
        self.dequeue(new_cluster.beam_id)

        if settings.detection.save_candidates:
            # Persist beam candidates
            self.save()

    def dequeue(self, beam_id):
        """
        Pop the last element from the queue if queue has reach maximum size

        :return:
        """
        if len(self.queue[beam_id]) > self._max_size:
            log.debug('Popping last candidate from queue')
            self.queue[beam_id].pop()

    def save(self):
        # get all clusters, across all queues, which have 'not saved'
        candidates_to_save = [candidate for queue in self.queue for candidate in queue if candidate.to_save]

        # get all clusters, across all queues, which have 'to delete'
        candidates_to_delete = [candidate for queue in self.queue for candidate in queue if candidate.to_delete]

        # delete old clusters that were merged
        if candidates_to_delete:
            self.repository.delete(candidates_to_delete)

        # add new clusters
        if candidates_to_save:
            self.repository.persist(candidates_to_save)

        log.info('Added %s candidates, Deleted %s candidates',
                 len(candidates_to_save),
                 len(candidates_to_delete))
