from typing import List, Dict, Any
import re

class ResponseVerifier:
    """Verify AI responses against source documents"""
    
    def __init__(self):
        pass
    
    def verify_response(self, response: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Verify response against source documents
        
        Args:
            response: Generated response
            sources: Source documents used
            
        Returns:
            Verification results with confidence score
        """
        if not sources:
            return {
                "confidence": 0,
                "supported_claims": 0,
                "total_claims": 0,
                "verification_status": "NO_SOURCES",
                "details": "No source documents available for verification"
            }
        
        # Extract claims from response (sentences)
        claims = self._extract_claims(response)
        
        if not claims:
            return {
                "confidence": 50,
                "supported_claims": 0,
                "total_claims": 0,
                "verification_status": "NO_CLAIMS",
                "details": "No verifiable claims found in response"
            }
        
        # Check each claim against sources
        supported_claims = 0
        verification_details = []
        
        for claim in claims:
            is_supported, matching_source = self._verify_claim(claim, sources)
            if is_supported:
                supported_claims += 1
                verification_details.append({
                    "claim": claim,
                    "supported": True,
                    "source": matching_source
                })
            else:
                verification_details.append({
                    "claim": claim,
                    "supported": False
                })
        
        # Calculate confidence score
        confidence = int((supported_claims / len(claims)) * 100) if claims else 0
        
        # Adjust confidence based on source similarity scores
        avg_similarity = sum(s.get('similarity_score', 0) for s in sources) / len(sources)
        confidence = int(confidence * (0.7 + 0.3 * avg_similarity))
        
        # Determine verification status
        if confidence >= 80:
            status = "HIGH_CONFIDENCE"
        elif confidence >= 50:
            status = "MEDIUM_CONFIDENCE"
        else:
            status = "LOW_CONFIDENCE"
        
        return {
            "confidence": confidence,
            "supported_claims": supported_claims,
            "total_claims": len(claims),
            "verification_status": status,
            "details": verification_details,
            "average_source_similarity": round(avg_similarity, 3)
        }
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract individual claims (sentences) from text"""
        # Split by sentence endings
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Minimum length for a claim
                claims.append(sentence)
        
        return claims
    
    def _verify_claim(self, claim: str, sources: List[Dict[str, Any]]) -> tuple[bool, str]:
        """
        Verify if a claim is supported by sources
        
        Args:
            claim: Claim to verify
            sources: Source documents
            
        Returns:
            Tuple of (is_supported, matching_source)
        """
        claim_lower = claim.lower()
        claim_words = set(claim_lower.split())
        
        best_match_score = 0
        best_source = ""
        
        for source in sources:
            source_text = source['text'].lower()
            source_words = set(source_text.split())
            
            # Calculate word overlap
            common_words = claim_words.intersection(source_words)
            
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been'}
            meaningful_common = common_words - stop_words
            
            if len(meaningful_common) > 0:
                # Calculate overlap score
                overlap_score = len(meaningful_common) / len(claim_words - stop_words) if claim_words - stop_words else 0
                
                if overlap_score > best_match_score:
                    best_match_score = overlap_score
                    best_source = source['document']
        
        # Threshold for considering a claim supported
        is_supported = best_match_score >= 0.3
        
        return is_supported, best_source
