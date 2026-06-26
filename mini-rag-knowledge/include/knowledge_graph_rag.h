#ifndef KNOWLEDGE_GRAPH_RAG_H
#define KNOWLEDGE_GRAPH_RAG_H

#include <stdbool.h>
#include <stddef.h>

/* ──────────────────────────────────────────────
   L8: Knowledge Graph Enhanced RAG

   GraphRAG extends vanilla RAG by extracting entities and
   relations from retrieved documents, constructing a local
   knowledge graph, and using graph structure to enhance
   retrieval relevance scoring.

   Key concepts:
   - Entity extraction: NER-like pattern matching
   - Relation extraction: co-occurrence based subject-predicate-object
   - Graph traversal: BFS for multi-hop evidence gathering
   - Graph-enhanced scoring: entity overlap + graph distance

   References:
   - Edge et al., "From Local to Global: A Graph RAG Approach"
     (Microsoft Research, 2024)
   - Wu et al., "Graph Neural Networks for NLP" (2019)
   ────────────────────────────────────────────── */

/* Entity types for knowledge graph nodes */
typedef enum {
    ENT_PERSON,
    ENT_ORGANIZATION,
    ENT_LOCATION,
    ENT_DATE,
    ENT_MONEY,
    ENT_PRODUCT,
    ENT_TECHNICAL_TERM,
    ENT_OTHER
} EntityType;

typedef enum {
    REL_WORKS_FOR,
    REL_LOCATED_IN,
    REL_FOUNDED,
    REL_PART_OF,
    REL_CAUSES,
    REL_PRODUCES,
    REL_DEPENDS_ON,
    REL_SIMILAR_TO,
    REL_MENTIONS,
    REL_OTHER
} RelationType;

/* Entity node in the knowledge graph */
typedef struct {
    size_t     id;
    char      *name;
    char      *normalized_name;
    EntityType type;
    size_t     first_char;
    size_t     last_char;
    size_t     doc_index;
    size_t     chunk_index;
    float      confidence;
} Entity;

/* Relation edge between two entities */
typedef struct {
    size_t       id;
    size_t       subject_id;
    size_t       object_id;
    RelationType type;
    char        *predicate;
    float        confidence;
    char        *evidence_text;
} Relation;

/* Knowledge Graph */
typedef struct {
    Entity    *entities;
    size_t     num_entities;
    size_t     entity_capacity;
    Relation  *relations;
    size_t     num_relations;
    size_t     relation_capacity;
} KnowledgeGraph;

/* Graph traversal path for multi-hop retrieval */
typedef struct {
    size_t  *entity_ids;
    size_t  *relation_ids;
    size_t   length;
} GraphPath;

typedef struct {
    GraphPath *paths;
    size_t     count;
    size_t     capacity;
} PathList;

/* Entity mention in document text */
typedef struct {
    char      *text;
    EntityType type;
    size_t     start_pos;
    size_t     end_pos;
    float      confidence;
} EntityMention;

typedef struct {
    EntityMention *mentions;
    size_t          count;
    size_t          capacity;
} EntityMentionList;

/* ──────────────────────────────────────────────
   Lifecycle
   ────────────────────────────────────────────── */
KnowledgeGraph*  kg_create(void);
void             kg_destroy(KnowledgeGraph *kg);

EntityMentionList*  entity_mention_list_create(void);
void                entity_mention_list_destroy(EntityMentionList *eml);

PathList*  pathlist_create(void);
void       pathlist_destroy(PathList *pl);

/* ──────────────────────────────────────────────
   Entity Extraction (NER via pattern matching)
   ────────────────────────────────────────────── */
EntityMentionList*  kg_extract_entities(const char *text, size_t text_len);

/* ──────────────────────────────────────────────
   Entity Normalization
   ────────────────────────────────────────────── */
char*  kg_normalize_entity_name(const char *name);

/* ──────────────────────────────────────────────
   Entity Linking (match mentions to KG entities)
   ────────────────────────────────────────────── */
size_t  kg_add_entity(KnowledgeGraph *kg, const EntityMention *mention,
                      size_t doc_idx, size_t chunk_idx);
size_t  kg_find_entity(const KnowledgeGraph *kg, const char *normalized_name);

/* ──────────────────────────────────────────────
   Relation Extraction (co-occurrence patterns)
   ────────────────────────────────────────────── */
void  kg_extract_relations(KnowledgeGraph *kg, const char *text,
                            size_t text_len, size_t doc_idx, size_t chunk_idx);

/* ──────────────────────────────────────────────
   Graph Construction from Document Corpus
   ────────────────────────────────────────────── */
KnowledgeGraph*  kg_build_from_chunks(const char **chunk_texts,
                                       const size_t *chunk_lens,
                                       size_t num_chunks,
                                       const size_t *doc_indices);

/* ──────────────────────────────────────────────
   Graph Traversal: multi-hop evidence gathering
   ────────────────────────────────────────────── */
PathList*  kg_find_paths(const KnowledgeGraph *kg, size_t start_entity,
                          size_t target_entity, size_t max_hops);

/* ──────────────────────────────────────────────
   Graph-enhanced Retrieval Scoring
   ────────────────────────────────────────────── */
float  kg_entity_overlap_score(const KnowledgeGraph *kg,
                                const size_t *query_entities,
                                size_t num_query_entities,
                                size_t chunk_idx);

float  kg_graph_distance_score(const KnowledgeGraph *kg,
                                const size_t *query_entities,
                                size_t num_query_entities,
                                size_t chunk_idx);

/* ──────────────────────────────────────────────
   Subgraph Construction for Context Window
   ────────────────────────────────────────────── */
KnowledgeGraph*  kg_extract_subgraph(const KnowledgeGraph *kg,
                                      const size_t *seed_entities,
                                      size_t num_seeds,
                                      size_t radius);

/* ──────────────────────────────────────────────
   Statistics
   ────────────────────────────────────────────── */
size_t  kg_entity_count(const KnowledgeGraph *kg);
size_t  kg_relation_count(const KnowledgeGraph *kg);
float   kg_density(const KnowledgeGraph *kg);

#endif /* KNOWLEDGE_GRAPH_RAG_H */