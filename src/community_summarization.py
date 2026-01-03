"""
Community Summarization using SSM (Mamba)
Generates hierarchical summaries for medical communities
Microsoft GraphRAG-inspired approach adapted for medical images
"""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from .ssm import SSMQueryProcessor
from .community_detection import Community

@dataclass
class CommunitySummary:
    """Summary of a medical community"""
    community_id: str
    level: int
    summary_text: str
    key_findings: List[str]
    statistics: Dict
    confidence: float

class CommunitySummarizer:
    """
    Generate summaries for hierarchical medical communities
    Uses SSM (Mamba) for text generation
    """
    
    def __init__(self, ssm_processor: Optional[SSMQueryProcessor] = None):
        """
        Initialize community summarizer
        
        Args:
            ssm_processor: SSM query processor for text generation
        """
        self.ssm_processor = ssm_processor or SSMQueryProcessor()
        self.summaries = {}
        
    def summarize_communities(self,
                             communities: Dict[str, Community],
                             node_metadata: Dict[str, Dict]) -> Dict[str, CommunitySummary]:
        """
        Generate summaries for all communities
        
        Args:
            communities: Dictionary of communities
            node_metadata: Metadata for all nodes
            
        Returns:
            Dictionary of community summaries
        """
        print("Generating community summaries...")
        
        # Sort communities by level (bottom-up for hierarchical summarization)
        sorted_communities = sorted(
            communities.items(),
            key=lambda x: x[1].level,
            reverse=True
        )
        
        for comm_id, community in sorted_communities:
            summary = self._generate_community_summary(
                community, node_metadata, communities
            )
            self.summaries[comm_id] = summary
            print(f"  ✓ {comm_id}: {summary.summary_text[:50]}...")
        
        print(f"✅ Generated {len(self.summaries)} community summaries")
        return self.summaries
    
    def _generate_community_summary(self,
                                   community: Community,
                                   node_metadata: Dict[str, Dict],
                                   all_communities: Dict[str, Community]) -> CommunitySummary:
        """Generate summary for a single community"""
        
        # Collect statistics
        stats = self._collect_community_statistics(community, node_metadata)
        
        # Generate summary based on level
        if community.level == 0:
            summary_text = self._summarize_global_community(community, stats)
        elif community.level == 1:
            summary_text = self._summarize_mid_level_community(community, stats, all_communities)
        else:  # Level 2
            summary_text = self._summarize_local_community(community, stats)
        
        # Extract key findings
        key_findings = self._extract_key_findings(community, stats)
        
        return CommunitySummary(
            community_id=community.id,
            level=community.level,
            summary_text=summary_text,
            key_findings=key_findings,
            statistics=stats,
            confidence=0.85  # Can be refined based on data quality
        )
    
    def _collect_community_statistics(self,
                                     community: Community,
                                     node_metadata: Dict[str, Dict]) -> Dict:
        """Collect statistical information about community"""
        stats = {
            'total_cases': len(community.member_nodes),
            'disease_type': community.disease_type,
            'class_distribution': {},
            'dataset_distribution': {}
        }
        
        for node_id in community.member_nodes:
            metadata = node_metadata.get(node_id, {})
            
            # Class distribution
            class_label = metadata.get('class_label', 'unknown')
            stats['class_distribution'][class_label] = stats['class_distribution'].get(class_label, 0) + 1
            
            # Dataset distribution
            dataset = metadata.get('dataset', 'unknown')
            stats['dataset_distribution'][dataset] = stats['dataset_distribution'].get(dataset, 0) + 1
        
        return stats
    
    def _summarize_global_community(self, community: Community, stats: Dict) -> str:
        """Generate global-level (Level 0) summary"""
        total_cases = stats['total_cases']
        disease = community.disease_type
        
        # Get class distribution
        class_dist = stats['class_distribution']
        class_summary = ", ".join([f"{count} {label}" for label, count in class_dist.items()])
        
        summary = (
            f"Global medical community for {disease} dataset containing {total_cases} cases. "
            f"Distribution: {class_summary}. "
            f"This community represents the overall disease category with diverse presentations."
        )
        
        return summary
    
    def _summarize_mid_level_community(self,
                                      community: Community,
                                      stats: Dict,
                                      all_communities: Dict[str, Community]) -> str:
        """Generate mid-level (Level 1) summary"""
        total_cases = stats['total_cases']
        disease = community.disease_type
        visual_features = community.metadata.get('visual_features', 'mixed patterns')
        
        # Get dominant class
        class_dist = stats['class_distribution']
        if class_dist:
            dominant_class = max(class_dist, key=class_dist.get)
            dominant_count = class_dist[dominant_class]
            dominant_pct = (dominant_count / total_cases) * 100
            
            summary = (
                f"Mid-level {disease} community with {total_cases} cases showing {visual_features}. "
                f"Predominantly {dominant_class} ({dominant_pct:.1f}%), "
                f"indicating a visually coherent subgroup within the broader {disease} category."
            )
        else:
            summary = (
                f"Mid-level {disease} community with {total_cases} cases showing {visual_features}. "
                f"Represents a visually distinct subgroup within the {disease} category."
            )
        
        return summary
    
    def _summarize_local_community(self, community: Community, stats: Dict) -> str:
        """Generate local-level (Level 2) summary"""
        total_cases = stats['total_cases']
        disease = community.disease_type
        class_label = community.metadata.get('class_label', 'unknown')
        
        summary = (
            f"Fine-grained community of {total_cases} {disease} cases "
            f"specifically classified as '{class_label}'. "
            f"This represents a homogeneous group with consistent clinical presentations, "
            f"ideal for detailed case-by-case analysis and comparison."
        )
        
        return summary
    
    def _extract_key_findings(self, community: Community, stats: Dict) -> List[str]:
        """Extract key findings from community"""
        findings = []
        
        # Finding 1: Size and scope
        findings.append(f"Community contains {stats['total_cases']} medical cases")
        
        # Finding 2: Disease focus
        findings.append(f"Focused on {community.disease_type} disease category")
        
        # Finding 3: Diversity or homogeneity
        class_dist = stats['class_distribution']
        if len(class_dist) == 1:
            findings.append("Highly homogeneous group (single class)")
        elif len(class_dist) > 3:
            findings.append("Diverse group with multiple class representations")
        
        # Finding 4: Hierarchical position
        if community.level == 0:
            findings.append("Global-level community for broad pattern analysis")
        elif community.level == 1:
            findings.append("Mid-level community for visual pattern grouping")
        else:
            findings.append("Local-level community for fine-grained case matching")
        
        return findings
    
    def get_hierarchical_summary(self, node_id: str, communities: Dict[str, Community]) -> str:
        """
        Get hierarchical summary for a specific node
        Combines summaries from all levels
        """
        summaries_text = []
        
        # Find node's communities at each level
        for level in [0, 1, 2]:
            for comm_id, community in communities.items():
                if community.level == level and node_id in community.member_nodes:
                    if comm_id in self.summaries:
                        summary = self.summaries[comm_id]
                        summaries_text.append(f"Level {level}: {summary.summary_text}")
                    break
        
        if summaries_text:
            return "\n".join(summaries_text)
        return "No community summaries available for this node."
    
    def generate_global_search_context(self,
                                      query: str,
                                      relevant_communities: List[str]) -> str:
        """
        Generate context for global search by combining community summaries
        Microsoft GraphRAG-style map-reduce
        """
        if not relevant_communities:
            return "No relevant communities found for this query."
        
        # Gather summaries
        context_parts = [f"Query: {query}\n"]
        context_parts.append("Relevant Medical Communities:\n")
        
        for idx, comm_id in enumerate(relevant_communities, 1):
            if comm_id in self.summaries:
                summary = self.summaries[comm_id]
                context_parts.append(f"\n{idx}. Community {comm_id} (Level {summary.level}):")
                context_parts.append(f"   {summary.summary_text}")
                context_parts.append(f"   Key Findings: {', '.join(summary.key_findings[:2])}")
        
        return "\n".join(context_parts)
    
    def generate_local_search_context(self,
                                     query: str,
                                     node_ids: List[str],
                                     node_metadata: Dict[str, Dict]) -> str:
        """
        Generate context for local search with specific cases
        """
        context_parts = [f"Query: {query}\n"]
        context_parts.append("Specific Medical Cases:\n")
        
        for idx, node_id in enumerate(node_ids[:10], 1):  # Limit to 10 cases
            metadata = node_metadata.get(node_id, {})
            context_parts.append(f"\n{idx}. Case {node_id}:")
            context_parts.append(f"   Disease: {metadata.get('dataset', 'unknown')}")
            context_parts.append(f"   Classification: {metadata.get('class_label', 'unknown')}")
            context_parts.append(f"   Path: {metadata.get('image_path', 'unknown')}")
        
        return "\n".join(context_parts)
    
    def export_summaries(self) -> Dict:
        """Export summaries for serialization"""
        export_data = {
            'summaries': {},
            'stats': {
                'total_summaries': len(self.summaries),
                'by_level': {0: 0, 1: 0, 2: 0}
            }
        }
        
        for comm_id, summary in self.summaries.items():
            export_data['summaries'][comm_id] = {
                'community_id': summary.community_id,
                'level': summary.level,
                'summary_text': summary.summary_text,
                'key_findings': summary.key_findings,
                'statistics': summary.statistics,
                'confidence': summary.confidence
            }
            export_data['stats']['by_level'][summary.level] += 1
        
        return export_data
    
    def print_summary_report(self):
        """Print summary report"""
        print("\n" + "=" * 60)
        print("COMMUNITY SUMMARIZATION REPORT")
        print("=" * 60)
        
        by_level = {0: [], 1: [], 2: []}
        for comm_id, summary in self.summaries.items():
            by_level[summary.level].append(summary)
        
        for level in [0, 1, 2]:
            summaries = by_level[level]
            if summaries:
                print(f"\nLevel {level} Summaries ({len(summaries)} communities):")
                for summary in summaries[:3]:  # Show first 3
                    print(f"\n  {summary.community_id}:")
                    print(f"    {summary.summary_text}")
                    print(f"    Cases: {summary.statistics['total_cases']}")
                if len(summaries) > 3:
                    print(f"\n  ... and {len(summaries) - 3} more summaries")
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    print("Community Summarization Module")
    print("Generates hierarchical summaries using SSM (Mamba)")
    print("Microsoft GraphRAG-inspired for medical images")

