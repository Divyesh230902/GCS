# CHIIR 2026 Paper Review Summary & Action Plan

## Paper Information
- **Submission ID**: 2359
- **Title**: Multimodal Knowledge Graph Representation with State-Space Models for Medical Image Retrieval
- **Decision**: **REJECTED** (Full Paper Track)
- **Acceptance Rate**: 27/77 (35%)

## Overall Decision Summary

The paper was **rejected** for the full paper track. While reviewers acknowledged the conceptual interest and promising quantitative results (Precision@5: 72.4%, MRR: 77.1%, Success Rate: 95.2%), the paper lacks sufficient empirical rigor, clarity, and scholarly grounding for acceptance.

---

## Key Issues Identified

### 1. **Evaluation & Dataset Concerns**
- **Small dataset**: Only 400 cases, uniformly sampled and balanced
- **Insufficient dataset description**: Dataset details are not adequately described
- **Unclear relevance grading**: 
  - Use of nDCG without explaining relevance levels
  - No clear criteria for relevance levels
  - Number of relevance levels not specified
- **Custom metric justification**: Custom metric (DCG without normalization) not well justified
- **Missing baseline comparisons**: Quantitative results and comparisons to strong baselines not clearly reported

### 2. **Presentation & Clarity Issues**
- **Dense and difficult to follow**: Manuscript is highly complex and dense
- **Unclear experimental design**: Research design for retrieval experiments requires clearer description
- **Lack of focus**: Need clearer focus on core contribution and more streamlined presentation
- **Metric definitions**: Experimental design and metric definitions require clearer explanation

### 3. **Scholarly Grounding Issues**
- **Heavy reliance on arXiv preprints**: 17 references are arXiv preprints (non-peer-reviewed)
- **Weak scholarly foundation**: Key components build on non-reviewed versions
- **Missing foundational citations**: Should cite Järvelin & Kekäläinen (2002) for DCG/nDCG framework
- **Figure licensing concerns**: Reproduced figures from arXiv sources without clear licensing/permission statements

### 4. **Missing Evaluation Components**
- **No user-centered evaluation**: Claims about interpretability and trust lack user-centered evaluation
- **Limited practical discussion**: 
  - Scalability concerns not addressed
  - Privacy considerations missing
  - Deployment constraints not discussed
  - Failure modes not explored
- **Sensitivity analysis**: Questions about sensitivity to embedding quality remain unanswered

---

## Reviewer Feedback Highlights

### Reviewer 1
- **Strengths**: Competitive retrieval performance, strong reliability
- **Weaknesses**: Small dataset, standard IR metrics don't quantify interpretability/trust, needs user-centric evaluation

### Reviewer 2
- **Strengths**: None explicitly stated
- **Weaknesses**: 
  - Highly complex and dense presentation
  - Unclear relevance criteria and metric justification
  - Heavy reliance on arXiv preprints
  - Figure licensing issues

### Reviewer 4
- **Strengths**: 
  - Practical and clinician-oriented approach
  - Clear, explainable groups
  - Traceable evidence and multiple retrieval modes
- **Weaknesses**: 
  - Lacks empirical detail
  - Questions about scalability, sensitivity, and practical adoption
  - Limited discussion of privacy, deployment, and failure modes

---

## Action Plan

### Priority 1: Address Evaluation & Dataset Issues

#### 1.1 Expand and Document Dataset
- [ ] Increase dataset size significantly (aim for 1000+ cases minimum)
- [ ] Provide detailed dataset description:
  - [ ] Source and collection methodology
  - [ ] Demographics and characteristics
  - [ ] Data preprocessing steps
  - [ ] Train/validation/test splits
- [ ] Document data balancing and sampling strategy

#### 1.2 Clarify Evaluation Metrics
- [ ] **Define relevance levels clearly**:
  - [ ] Specify number of relevance levels (e.g., 0-4 scale)
  - [ ] Provide clear criteria for each relevance level
  - [ ] Include examples of relevance judgments
- [ ] **Justify custom metric**:
  - [ ] Explain why DCG without normalization is used
  - [ ] Compare against standard nDCG
  - [ ] Provide theoretical justification
- [ ] **Cite foundational work**:
  - [ ] Add citation to Järvelin & Kekäläinen (2002) for DCG/nDCG
  - [ ] Reference other relevant IR evaluation literature

#### 1.3 Strengthen Baseline Comparisons
- [ ] Include comprehensive baseline comparisons:
  - [ ] Standard CLIP-based retrieval
  - [ ] Graph-based retrieval methods
  - [ ] State-of-the-art medical image retrieval systems
- [ ] Report statistical significance tests
- [ ] Provide detailed quantitative results tables

### Priority 2: Improve Presentation & Clarity

#### 2.1 Restructure Paper
- [ ] **Streamline presentation**:
  - [ ] Identify and emphasize core contribution clearly
  - [ ] Reduce density by moving details to appendices
  - [ ] Improve logical flow and organization
- [ ] **Clarify experimental design**:
  - [ ] Add detailed methodology section
  - [ ] Explain experimental setup step-by-step
  - [ ] Include experimental protocol diagram

#### 2.2 Improve Metric Definitions
- [ ] Add clear definitions section for all metrics
- [ ] Provide examples of metric calculations
- [ ] Explain why each metric is appropriate for the task

### Priority 3: Strengthen Scholarly Foundation

#### 3.1 Replace arXiv References
- [ ] **Audit all references**:
  - [ ] Identify all 17 arXiv preprints
  - [ ] Check if peer-reviewed versions exist
  - [ ] Replace with peer-reviewed versions where available
- [ ] **For remaining preprints**:
  - [ ] Justify why preprint is necessary
  - [ ] Note publication status in text
  - [ ] Consider alternative peer-reviewed sources

#### 3.2 Address Figure Licensing
- [ ] **For all reproduced figures**:
  - [ ] Check licensing/permissions for each figure
  - [ ] Obtain proper permissions or redraw figures
  - [ ] Add proper attribution and licensing information
  - [ ] Consider creating original figures instead

#### 3.3 Expand Literature Review
- [ ] Add more peer-reviewed IR literature
- [ ] Better situate contribution within established research
- [ ] Include recent work on medical image retrieval
- [ ] Reference work on interpretable retrieval systems

### Priority 4: Add Missing Evaluation Components

#### 4.1 User-Centered Evaluation
- [ ] **Design user study**:
  - [ ] Recruit domain experts (clinicians/radiologists)
  - [ ] Design tasks to evaluate interpretability
  - [ ] Measure trust and usability
  - [ ] Collect qualitative feedback
- [ ] **Report user study results**:
  - [ ] Include user satisfaction metrics
  - [ ] Report on interpretability assessment
  - [ ] Document trust measures

#### 4.2 Practical Considerations
- [ ] **Address scalability**:
  - [ ] Analyze computational complexity
  - [ ] Report on large-scale experiments
  - [ ] Discuss scalability limitations
- [ ] **Privacy & deployment**:
  - [ ] Discuss HIPAA/medical data privacy considerations
  - [ ] Address deployment constraints
  - [ ] Consider federated learning approaches if applicable
- [ ] **Failure analysis**:
  - [ ] Analyze failure cases
  - [ ] Identify common error patterns
  - [ ] Discuss limitations and edge cases

#### 4.3 Sensitivity Analysis
- [ ] Evaluate sensitivity to embedding quality
- [ ] Test robustness to different embedding models
- [ ] Analyze impact of graph structure variations

### Priority 5: Additional Improvements

#### 5.1 Strengthen Technical Contribution
- [ ] More clearly articulate novelty vs. existing work
- [ ] Provide theoretical analysis where possible
- [ ] Include ablation studies for each component

#### 5.2 Improve Reproducibility
- [ ] Release code and data (if possible)
- [ ] Provide detailed hyperparameter settings
- [ ] Include implementation details
- [ ] Add reproducibility checklist

#### 5.3 Future Work
- [ ] Expand discussion of limitations
- [ ] Provide concrete directions for future research
- [ ] Address reviewer concerns in future work section

---

## Timeline Recommendations

### Phase 1: Critical Fixes (2-3 months)
- Address dataset and evaluation issues
- Improve clarity and presentation
- Replace arXiv references

### Phase 2: Enhanced Evaluation (3-4 months)
- Conduct user-centered evaluation
- Add baseline comparisons
- Perform sensitivity analysis

### Phase 3: Final Polish (1-2 months)
- Address all remaining concerns
- Final proofreading and editing
- Prepare for resubmission

---

## Target Venues for Resubmission

Consider resubmitting to:
- **ACM CHIIR** (after addressing all concerns)
- **SIGIR** (Information Retrieval)
- **EMNLP/ACL** (if emphasizing NLP aspects)
- **Medical Informatics journals** (e.g., JAMIA, JBI)
- **Medical Imaging conferences** (e.g., MICCAI, SPIE Medical Imaging)

---

## Notes

- The reviewers acknowledged the work is "conceptually interesting" and results are "encouraging"
- The core idea (Graph-CLIP-State) has merit but needs stronger empirical validation
- Focus on making the paper more accessible and rigorous
- Consider breaking into multiple papers if the scope is too broad

