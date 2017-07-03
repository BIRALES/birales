from pybirales.modules.detection.repository import BeamCandidateRepository
from pybirales.base import settings
import logging as log
from multiprocessing.queues import Queue


class BeamCandidatesQueue:
    def __init__(self, n_beams):
        self.queue = [[] for _ in range(n_beams)]
        self._max_size = 20
        self.repository = None

        self.candidates_to_delete = []

        # super(BeamCandidatesQueue, self).__init__(maxsize=self._max_size)
        # self.repository = BeamCandidateRepository()

    def set_repository(self, repository):
        self.repository = repository

    def enqueue(self, new_cluster):
        # Check if beam candidate is similar to candidate that was already added to the queue
        # all_clusters_compared = False
        # beam_queue = self.queue[new_cluster.beam_id]
        # n = 0
        # while True:
        #     queue_length = len(self.queue[new_cluster.beam_id])
        #
        #     if n == queue_length:
        #         break

        beam_queue = self.queue[new_cluster.beam_id]
        # log.warning('Queue %s length is %s (before merging)', new_cluster.beam_id, len(beam_queue))
        for i, old_cluster in enumerate(beam_queue):
            if old_cluster.to_delete:
                continue

            if new_cluster.is_similar_to(old_cluster, threshold=0.2):
                # log.debug('Merging clusters [%s -> %s] in beam_queue %s (length: %s)',
                #           len(old_cluster.merge(new_cluster).time_data), len(old_cluster.time_data),
                          # new_cluster.beam_id, len(beam_queue))
                # mark old cluster as 'to delete'
                old_cluster.delete()
                # beam_queue.remove(old_cluster)
                # merge new cluster with the old one and add to the queue
                # beam_queue.insert(0, old_cluster.merge(new_cluster))
                # log.warning('Inserted a merged cluster of size %s (from %s)',
                #             len(old_cluster.merge(new_cluster).time_data), len(old_cluster.time_data))
                self.enqueue(old_cluster.merge(new_cluster))
                # Exit the loop once we added the new cluster to the queue
                break
        else:
            # If we didn't merge, add the candidate to the queue
            beam_queue.insert(0, new_cluster)
            # log.warning('Inserted a cluster of size %s', len(new_cluster.time_data))

        # log.warning('Queue %s length is %s (after merging)', new_cluster.beam_id, len(beam_queue))
        # log.debug('Enqueuing took %1.3f s', time.time() - t1)
        # Pop the last element if queue is full
        self.dequeue(new_cluster.beam_id)

        if settings.detection.save_candidates:
            # Persist beam candidates
            self.save()

        # log.warning('Queue %s length is %s (after merging + deletion)', new_cluster.beam_id,
        #             len(self.queue[new_cluster.beam_id]))

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

        # Garbage collect - remove candidates marked for deletion from queue
        for q, queue in enumerate(self.queue):
            self.queue[q] = [candidate for candidate in queue if not candidate.to_delete]
