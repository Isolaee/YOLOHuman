'''
Docstring for facial_recognition.face_matches
Module for matching and clustering faces based on embedding similarity.
'''

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field


@dataclass
class FaceMatch:
    """Represents a face detection with its source information."""
    image_path: str
    person_id: int
    face_id: int
    embedding: np.ndarray
    features: Dict
    
    @property
    def key(self) -> str:
        """Unique key for this face."""
        return f"{self.image_path}:p{self.person_id}:f{self.face_id}"


@dataclass 
class PersonCluster:
    """A cluster of faces belonging to the same person."""
    cluster_id: int
    faces: List[FaceMatch] = field(default_factory=list)
    
    @property
    def representative_embedding(self) -> Optional[np.ndarray]:
        """Get average embedding for this cluster."""
        if not self.faces:
            return None
        embeddings = [f.embedding for f in self.faces if f.embedding is not None]
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0)
    
    @property
    def image_count(self) -> int:
        """Number of unique images this person appears in."""
        return len(set(f.image_path for f in self.faces))
    
    def get_summary(self) -> Dict:
        """Get summary info about this cluster."""
        ages = [f.features.get('age') for f in self.faces if f.features.get('age')]
        genders = [f.features.get('gender') for f in self.faces if f.features.get('gender')]
        
        return {
            'cluster_id': self.cluster_id,
            'face_count': len(self.faces),
            'image_count': self.image_count,
            'avg_age': int(np.mean(ages)) if ages else None,
            'gender': max(set(genders), key=genders.count) if genders else None,
            'images': list(set(f.image_path for f in self.faces))
        }


class FaceMatcher:
    """
    Match and cluster faces across multiple images.
    Uses cosine similarity on InsightFace embeddings.
    """
    
    def __init__(self, similarity_threshold: float = 0.4):
        """
        Initialize the face matcher.
        
        Args:
            similarity_threshold: Min similarity to consider faces as same person (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.faces: List[FaceMatch] = []
        self.clusters: List[PersonCluster] = []
    
    @staticmethod
    def compute_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two face embeddings.
        
        Args:
            embedding1: First face embedding (512-dim, normalized)
            embedding2: Second face embedding (512-dim, normalized)
            
        Returns:
            Similarity score (0-1, higher = more similar)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        # InsightFace embeddings are L2-normalized, so dot product = cosine similarity
        return float(np.dot(embedding1, embedding2))
    
    def add_face(self, 
                 image_path: str,
                 person_id: int, 
                 face_id: int,
                 embedding: np.ndarray,
                 features: Dict):
        """
        Add a face to the matcher.
        
        Args:
            image_path: Source image path
            person_id: Person index in the image
            face_id: Face index within the person detection
            embedding: Face embedding vector
            features: Face features dict
        """
        face = FaceMatch(
            image_path=image_path,
            person_id=person_id,
            face_id=face_id,
            embedding=embedding,
            features=features
        )
        self.faces.append(face)
    
    def add_from_results(self, all_results: Dict[str, List[Dict]]):
        """
        Add faces from PersonFaceAnalyzer results.
        
        Args:
            all_results: Dict mapping image_path -> list of person results
        """
        for image_path, persons in all_results.items():
            for person in persons:
                for face in person['faces']:
                    if face.get('embedding') is not None:
                        self.add_face(
                            image_path=image_path,
                            person_id=person['person_id'],
                            face_id=face['face_id'],
                            embedding=face['embedding'],
                            features=face
                        )
    
    def find_matches(self, query_embedding: np.ndarray) -> List[Tuple[FaceMatch, float]]:
        """
        Find all faces matching a query embedding.
        
        Args:
            query_embedding: Face embedding to search for
            
        Returns:
            List of (FaceMatch, similarity) tuples, sorted by similarity
        """
        matches = []
        for face in self.faces:
            if face.embedding is None:
                continue
            similarity = self.compute_similarity(query_embedding, face.embedding)
            if similarity >= self.similarity_threshold:
                matches.append((face, similarity))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def cluster_faces(self) -> List[PersonCluster]:
        """
        Cluster all faces by identity using greedy clustering.
        Faces with similarity >= threshold are grouped together.
        
        Returns:
            List of PersonCluster objects
        """
        if not self.faces:
            return []
        
        # Filter faces with valid embeddings
        valid_faces = [f for f in self.faces if f.embedding is not None]
        
        if not valid_faces:
            return []
        
        assigned = set()
        clusters = []
        cluster_id = 1
        
        for i, face in enumerate(valid_faces):
            if i in assigned:
                continue
            
            # Start new cluster with this face
            cluster = PersonCluster(cluster_id=cluster_id)
            cluster.faces.append(face)
            assigned.add(i)
            
            # Find all similar faces
            for j in range(i + 1, len(valid_faces)):
                if j in assigned:
                    continue
                
                # Check similarity against all faces in cluster (average linkage idea)
                max_similarity = max(
                    self.compute_similarity(face.embedding, valid_faces[j].embedding)
                    for face in cluster.faces
                )
                
                if max_similarity >= self.similarity_threshold:
                    cluster.faces.append(valid_faces[j])
                    assigned.add(j)
            
            clusters.append(cluster)
            cluster_id += 1
        
        self.clusters = clusters
        return clusters
    
    def get_person_appearances(self) -> Dict[int, List[str]]:
        """
        Get which images each unique person appears in.
        
        Returns:
            Dict mapping cluster_id -> list of image paths
        """
        if not self.clusters:
            self.cluster_faces()
        
        return {
            cluster.cluster_id: list(set(f.image_path for f in cluster.faces))
            for cluster in self.clusters
        }
    
    def find_same_person_across_images(self) -> List[PersonCluster]:
        """
        Find persons that appear in multiple images.
        
        Returns:
            List of PersonClusters that span multiple images
        """
        if not self.clusters:
            self.cluster_faces()
        
        return [c for c in self.clusters if c.image_count > 1]
    
    def build_similarity_matrix(self) -> Tuple[np.ndarray, List[FaceMatch]]:
        """
        Build a similarity matrix for all faces.
        
        Returns:
            Tuple of (similarity_matrix, face_list)
        """
        valid_faces = [f for f in self.faces if f.embedding is not None]
        n = len(valid_faces)
        
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                sim = self.compute_similarity(valid_faces[i].embedding, valid_faces[j].embedding)
                matrix[i, j] = sim
                matrix[j, i] = sim
        
        return matrix, valid_faces
    
    def print_clusters(self):
        """Print cluster information."""
        if not self.clusters:
            self.cluster_faces()
        
        print(f"\nFound {len(self.clusters)} unique person(s):\n")
        
        for cluster in self.clusters:
            summary = cluster.get_summary()
            print(f"Person {summary['cluster_id']}:")
            print(f"  Appears in {summary['image_count']} image(s), {summary['face_count']} detection(s)")
            
            if summary['avg_age']:
                print(f"  Average age: {summary['avg_age']}, Gender: {summary['gender']}")
            
            print(f"  Images:")
            for face in cluster.faces:
                import os
                img_name = os.path.basename(face.image_path)
                age = face.features.get('age', '?')
                gender = face.features.get('gender', '?')
                score = face.features.get('det_score', 0)
                print(f"    - {img_name} (Person {face.person_id}, Face {face.face_id}) "
                      f"[{gender}, {age}yo, score:{score:.2f}]")
            print()
    
    def get_cross_image_matches(self) -> List[Dict]:
        """
        Get detailed information about persons appearing in multiple images.
        
        Returns:
            List of dicts with match information
        """
        multi_image_clusters = self.find_same_person_across_images()
        
        results = []
        for cluster in multi_image_clusters:
            match_info = {
                'person_id': cluster.cluster_id,
                'summary': cluster.get_summary(),
                'appearances': []
            }
            
            for face in cluster.faces:
                match_info['appearances'].append({
                    'image': face.image_path,
                    'person_id': face.person_id,
                    'face_id': face.face_id,
                    'age': face.features.get('age'),
                    'gender': face.features.get('gender'),
                    'det_score': face.features.get('det_score')
                })
            
            results.append(match_info)
        
        return results
