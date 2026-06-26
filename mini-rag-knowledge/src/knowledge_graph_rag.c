#include "knowledge_graph_rag.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>

/* ──────────────────────────────────────────────
   Internal helpers
   ────────────────────────────────────────────── */
static void *safe_alloc(size_t n) {
    void *p = calloc(n, 1);
    if (!p) { fprintf(stderr, "OOM: kg alloc %zu\n", n); exit(1); }
    return p;
}

static char *safe_strdup(const char *s) {
    if (!s) return NULL;
    size_t n = strlen(s);
    char *d = safe_alloc(n + 1);
    memcpy(d, s, n);
    return d;
}

static bool is_uppercase(char c) {
    return c >= 'A' && c <= 'Z';
}

static bool is_name_char(char c) {
    return isalnum((unsigned char)c) || c == '.' || c == '-' || c == '_' || c == '&';
}

static bool is_person_title(const char *word, size_t len) {
    const char *titles[] = {"Mr", "Mrs", "Ms", "Dr", "Prof", "Sir", "Lord", "President", "CEO", "CTO", "CFO"};
    for (size_t i = 0; i < sizeof(titles)/sizeof(titles[0]); i++) {
        size_t tl = strlen(titles[i]);
        if (len == tl && strncmp(word, titles[i], tl) == 0) return true;
    }
    return false;
}

static bool is_org_suffix(const char *word, size_t len) {
    const char *suffixes[] = {"Inc", "Corp", "LLC", "Ltd", "GmbH", "AG", "SA", "PLC", "Group", "University", "Institute", "Laboratory", "Department"};
    for (size_t i = 0; i < sizeof(suffixes)/sizeof(suffixes[0]); i++) {
        size_t sl = strlen(suffixes[i]);
        if (len == sl && strncmp(word, suffixes[i], sl) == 0) return true;
    }
    return false;
}

/* Word tokenization helpers */
static size_t next_word(const char *text, size_t len, size_t start,
                         size_t *word_start, size_t *word_end) {
    while (start < len && isspace((unsigned char)text[start])) start++;
    if (start >= len) return len;
    *word_start = start;
    while (start < len && is_name_char(text[start])) start++;
    *word_end = start;
    return start;
}

/* ──────────────────────────────────────────────
   KnowledgeGraph Lifecycle
   ────────────────────────────────────────────── */
KnowledgeGraph* kg_create(void) {
    KnowledgeGraph *kg = safe_alloc(sizeof(KnowledgeGraph));
    kg->entity_capacity   = 256;
    kg->relation_capacity = 256;
    kg->entities   = safe_alloc(kg->entity_capacity * sizeof(Entity));
    kg->relations  = safe_alloc(kg->relation_capacity * sizeof(Relation));
    kg->num_entities  = 0;
    kg->num_relations = 0;
    return kg;
}

void kg_destroy(KnowledgeGraph *kg) {
    if (!kg) return;
    for (size_t i = 0; i < kg->num_entities; i++) {
        free(kg->entities[i].name);
        free(kg->entities[i].normalized_name);
    }
    for (size_t i = 0; i < kg->num_relations; i++) {
        free(kg->relations[i].predicate);
        free(kg->relations[i].evidence_text);
    }
    free(kg->entities);
    free(kg->relations);
    free(kg);
}

/* ──────────────────────────────────────────────
   Entity Mention List
   ────────────────────────────────────────────── */
EntityMentionList* entity_mention_list_create(void) {
    EntityMentionList *eml = safe_alloc(sizeof(EntityMentionList));
    eml->capacity = 64;
    eml->mentions = safe_alloc(eml->capacity * sizeof(EntityMention));
    eml->count    = 0;
    return eml;
}

void entity_mention_list_destroy(EntityMentionList *eml) {
    if (!eml) return;
    for (size_t i = 0; i < eml->count; i++) {
        free(eml->mentions[i].text);
    }
    free(eml->mentions);
    free(eml);
}

static void eml_add(EntityMentionList *eml, const char *text,
                     EntityType type, size_t start, size_t end, float conf) {
    if (eml->count >= eml->capacity) {
        eml->capacity *= 2;
        eml->mentions = realloc(eml->mentions,
                                 eml->capacity * sizeof(EntityMention));
        if (!eml->mentions) { fprintf(stderr, "EntityMention OOM\n"); exit(1); }
    }
    EntityMention *m = &eml->mentions[eml->count];
    size_t tl = end - start;
    m->text      = safe_alloc(tl + 1);
    memcpy(m->text, text + start, tl);
    m->text[tl]  = '\0';
    m->type      = type;
    m->start_pos = start;
    m->end_pos   = end;
    m->confidence = conf;
    eml->count++;
}

/* ──────────────────────────────────────────────
   PathList Lifecycle
   ────────────────────────────────────────────── */
PathList* pathlist_create(void) {
    PathList *pl = safe_alloc(sizeof(PathList));
    pl->capacity = 32;
    pl->paths    = safe_alloc(pl->capacity * sizeof(GraphPath));
    pl->count    = 0;
    return pl;
}

void pathlist_destroy(PathList *pl) {
    if (!pl) return;
    for (size_t i = 0; i < pl->count; i++) {
        free(pl->paths[i].entity_ids);
        free(pl->paths[i].relation_ids);
    }
    free(pl->paths);
    free(pl);
}

/* ──────────────────────────────────────────────
   Entity Extraction (rule-based NER)

   Detects:
   - PERSON: capitalized name sequences with optional title prefix
   - ORGANIZATION: capitalized names ending with org suffixes
   - LOCATION: capitalized terms after "in/at/near" prepositions
   - DATE: YYYY-MM-DD or Month Day, Year patterns
   - TECHNICAL_TERM: CamelCase or snake_case identifiers
   ────────────────────────────────────────────── */
EntityMentionList* kg_extract_entities(const char *text, size_t text_len) {
    EntityMentionList *eml = entity_mention_list_create();
    if (!text || text_len == 0) return eml;

    size_t pos = 0;
    while (pos < text_len) {
        /* Skip non-name-start characters */
        while (pos < text_len && !is_uppercase(text[pos]) && !isdigit((unsigned char)text[pos])) {
            pos++;
        }
        if (pos >= text_len) break;

        /* Check for date pattern: 4 digits, or Month Name */
        bool is_date = false;
        if (isdigit((unsigned char)text[pos])) {
            size_t num_start = pos;
            while (pos < text_len && isdigit((unsigned char)text[pos])) pos++;
            if (pos - num_start == 4) {
                is_date = true;
            } else if (pos < text_len && text[pos] == '/') {
                pos++;
                while (pos < text_len && isdigit((unsigned char)text[pos])) pos++;
                if (pos < text_len && text[pos] == '/') {
                    pos++;
                    while (pos < text_len && isdigit((unsigned char)text[pos])) pos++;
                    is_date = true;
                }
            }
            if (is_date) {
                eml_add(eml, text, ENT_DATE, num_start, pos, 0.9f);
                continue;
            }
            continue; /* Not a date, reset */
        }

        /* Capitalized word sequence: potential named entity */
        size_t ent_start = pos;
        size_t ws = pos, we = pos;
        bool has_title = false;

        /* Check if first word is a person title */
        pos = next_word(text, text_len, pos, &ws, &we);
        if (is_person_title(text + ws, we - ws)) {
            has_title = true;
            pos = next_word(text, text_len, we + 1, &ws, &we);
        }

        /* Collect capitalized words */
        size_t cap_count = 0;
        size_t last_we = we;
        while (we < text_len && is_uppercase(text[ws]) && cap_count < 8) {
            cap_count++;
            last_we = we;
            size_t next_pos = next_word(text, text_len, we, &ws, &we);
            if (ws < text_len && !is_uppercase(text[ws])) {
                /* Check for org suffix */
                if (is_org_suffix(text + ws, we - ws)) {
                    last_we = we;
                }
                break;
            }
            pos = next_pos;
        }

        if (cap_count > 0) {
            EntityType etype = ENT_OTHER;
            if (has_title) {
                etype = ENT_PERSON;
            } else if (cap_count == 1 && is_uppercase(text[ent_start])) {
                etype = ENT_LOCATION; /* Single capitalized = likely location */
            }
            /* Check suffix for org determination */
            for (size_t check = ent_start; check < last_we; check++) {
                if (is_org_suffix(text + check, last_we - check)) {
                    etype = ENT_ORGANIZATION;
                    break;
                }
            }
            if (etype == ENT_OTHER && cap_count > 1) {
                etype = ENT_PERSON;
            }
            eml_add(eml, text, etype, ent_start, last_we,
                    cap_count >= 2 ? 0.85f : 0.6f);
        }
        pos = last_we;
    }

    return eml;
}

/* ──────────────────────────────────────────────
   Entity Name Normalization

   Lowercase, trim whitespace, remove punctuation.
   ────────────────────────────────────────────── */
char* kg_normalize_entity_name(const char *name) {
    if (!name) return NULL;
    size_t nl = strlen(name);
    char *norm = safe_alloc(nl + 1);
    size_t j = 0;
    for (size_t i = 0; i < nl; i++) {
        char c = name[i];
        if (isalnum((unsigned char)c) || c == ' ') {
            norm[j++] = (char)tolower((unsigned char)c);
        }
    }
    norm[j] = '\0';
    /* Trim trailing spaces */
    while (j > 0 && norm[j-1] == ' ') { j--; norm[j] = '\0'; }
    return norm;
}

/* ──────────────────────────────────────────────
   Entity Add / Find (with deduplication by normalized name)
   ────────────────────────────────────────────── */
size_t kg_add_entity(KnowledgeGraph *kg, const EntityMention *mention,
                      size_t doc_idx, size_t chunk_idx) {
    if (!kg || !mention) return (size_t)-1;

    char *norm = kg_normalize_entity_name(mention->text);

    /* Check for duplicates */
    for (size_t i = 0; i < kg->num_entities; i++) {
        if (kg->entities[i].normalized_name &&
            strcmp(kg->entities[i].normalized_name, norm) == 0) {
            /* Update confidence if higher */
            if (mention->confidence > kg->entities[i].confidence) {
                kg->entities[i].confidence = mention->confidence;
                kg->entities[i].type = mention->type;
            }
            free(norm);
            return i;
        }
    }

    /* Grow if needed */
    if (kg->num_entities >= kg->entity_capacity) {
        kg->entity_capacity *= 2;
        kg->entities = realloc(kg->entities,
                                kg->entity_capacity * sizeof(Entity));
        if (!kg->entities) { fprintf(stderr, "KG entity OOM\n"); exit(1); }
    }

    Entity *e = &kg->entities[kg->num_entities];
    e->id              = kg->num_entities;
    e->name            = safe_strdup(mention->text);
    e->normalized_name = norm;
    e->type            = mention->type;
    e->first_char      = mention->start_pos;
    e->last_char       = mention->end_pos;
    e->doc_index       = doc_idx;
    e->chunk_index     = chunk_idx;
    e->confidence      = mention->confidence;
    kg->num_entities++;
    return e->id;
}

size_t kg_find_entity(const KnowledgeGraph *kg, const char *normalized_name) {
    if (!kg || !normalized_name) return (size_t)-1;
    char *norm = kg_normalize_entity_name(normalized_name);
    for (size_t i = 0; i < kg->num_entities; i++) {
        if (kg->entities[i].normalized_name &&
            strcmp(kg->entities[i].normalized_name, norm) == 0) {
            free(norm);
            return i;
        }
    }
    free(norm);
    return (size_t)-1;
}

/* ──────────────────────────────────────────────
   Relation Extraction

   Uses pattern-based extraction:
   - "[Entity] works for [Entity]" → REL_WORKS_FOR
   - "[Entity] is located in [Entity]" → REL_LOCATED_IN
   - "[Entity] founded [Entity]" → REL_FOUNDED
   - "[Entity] causes [Entity]" → REL_CAUSES
   - "[Entity] depends on [Entity]" → REL_DEPENDS_ON

   Also creates co-occurrence relations for entities
   appearing in the same chunk (REL_MENTIONS).
   ────────────────────────────────────────────── */
static void add_relation(KnowledgeGraph *kg, size_t subj, size_t obj,
                          RelationType type, const char *predicate,
                          float conf, const char *evidence) {
    /* Avoid self-loops */
    if (subj == obj || subj >= kg->num_entities || obj >= kg->num_entities)
        return;

    /* Check for duplicate */
    for (size_t i = 0; i < kg->num_relations; i++) {
        if (kg->relations[i].subject_id == subj &&
            kg->relations[i].object_id == obj &&
            kg->relations[i].type == type) {
            return;
        }
    }

    if (kg->num_relations >= kg->relation_capacity) {
        kg->relation_capacity *= 2;
        kg->relations = realloc(kg->relations,
                                 kg->relation_capacity * sizeof(Relation));
        if (!kg->relations) { fprintf(stderr, "KG relation OOM\n"); exit(1); }
    }

    Relation *r = &kg->relations[kg->num_relations];
    r->id          = kg->num_relations;
    r->subject_id  = subj;
    r->object_id   = obj;
    r->type        = type;
    r->predicate   = predicate ? safe_strdup(predicate) : NULL;
    r->confidence  = conf;
    r->evidence_text = evidence ? safe_strdup(evidence) : NULL;
    kg->num_relations++;
}

/*
 * Simple relation patterns to detect in text
 */
typedef struct {
    const char *pattern;
    RelationType type;
    float confidence;
} RelPattern;

static const RelPattern REL_PATTERNS[] = {
    {" works for ",     REL_WORKS_FOR,   0.8f},
    {" employed by ",   REL_WORKS_FOR,   0.8f},
    {" located in ",    REL_LOCATED_IN,  0.9f},
    {" based in ",      REL_LOCATED_IN,  0.85f},
    {" headquartered ", REL_LOCATED_IN,  0.85f},
    {" founded ",       REL_FOUNDED,     0.9f},
    {" created by ",    REL_FOUNDED,     0.85f},
    {" causes ",        REL_CAUSES,      0.75f},
    {" leads to ",      REL_CAUSES,      0.75f},
    {" results in ",    REL_CAUSES,      0.7f},
    {" depends on ",    REL_DEPENDS_ON,  0.8f},
    {" requires ",      REL_DEPENDS_ON,  0.7f},
    {" part of ",       REL_PART_OF,     0.85f},
    {" belongs to ",    REL_PART_OF,     0.8f},
    {" similar to ",    REL_SIMILAR_TO,  0.6f},
    {" produces ",      REL_PRODUCES,    0.7f},
    {NULL, REL_OTHER, 0.0f}
};

void kg_extract_relations(KnowledgeGraph *kg, const char *text,
                           size_t text_len, size_t doc_idx, size_t chunk_idx) {
    (void)doc_idx;
    if (!kg || !text || text_len < 3) return;

    /* First, create co-occurrence relations for entities in this chunk */
    size_t chunk_entities[64];
    size_t chunk_ent_count = 0;
    for (size_t i = 0; i < kg->num_entities && chunk_ent_count < 64; i++) {
        if (kg->entities[i].chunk_index == chunk_idx) {
            chunk_entities[chunk_ent_count++] = i;
        }
    }

    /* Co-occurrence: every pair of entities in the same chunk */
    for (size_t i = 0; i < chunk_ent_count; i++) {
        for (size_t j = i + 1; j < chunk_ent_count; j++) {
            add_relation(kg, chunk_entities[i], chunk_entities[j],
                         REL_MENTIONS, "co-occurs with", 0.5f, text);
        }
    }

    /* Pattern-based relation extraction */
    for (const RelPattern *rp = REL_PATTERNS; rp->pattern; rp++) {
        size_t plen = strlen(rp->pattern);
        if (plen > text_len) continue;

        for (size_t pos = 0; pos <= text_len - plen; pos++) {
            if (strncmp(text + pos, rp->pattern, plen) != 0) continue;

            /* Find entity before the pattern */
            for (size_t i = 0; i < chunk_ent_count; i++) {
                Entity *subj = &kg->entities[chunk_entities[i]];
                if (subj->last_char <= pos + 1 && subj->last_char > pos - 50) {
                    /* Find entity after the pattern */
                    size_t after_pos = pos + plen;
                    for (size_t j = 0; j < chunk_ent_count; j++) {
                        if (i == j) continue;
                        Entity *obj = &kg->entities[chunk_entities[j]];
                        if (obj->first_char >= after_pos &&
                            obj->first_char < after_pos + 100) {
                            add_relation(kg, subj->id, obj->id,
                                         rp->type, rp->pattern,
                                         rp->confidence, text + pos);
                            break; /* One relation per pattern instance */
                        }
                    }
                    break; /* One subject per pattern instance */
                }
            }
        }
    }
}

/* ──────────────────────────────────────────────
   Knowledge Graph Construction from Chunks

   Pipeline: Extract entities → Deduplicate → Extract relations
   ────────────────────────────────────────────── */
KnowledgeGraph* kg_build_from_chunks(const char **chunk_texts,
                                       const size_t *chunk_lens,
                                       size_t num_chunks,
                                       const size_t *doc_indices) {
    KnowledgeGraph *kg = kg_create();
    if (!chunk_texts || num_chunks == 0) return kg;

    /* Phase 1: Extract entities from all chunks */
    for (size_t ci = 0; ci < num_chunks; ci++) {
        size_t di = doc_indices ? doc_indices[ci] : 0;
        EntityMentionList *eml = kg_extract_entities(chunk_texts[ci],
                                                      chunk_lens[ci]);
        for (size_t mi = 0; mi < eml->count; mi++) {
            kg_add_entity(kg, &eml->mentions[mi], di, ci);
        }
        entity_mention_list_destroy(eml);
    }

    /* Phase 2: Extract relations from all chunks */
    for (size_t ci = 0; ci < num_chunks; ci++) {
        size_t di = doc_indices ? doc_indices[ci] : 0;
        kg_extract_relations(kg, chunk_texts[ci], chunk_lens[ci], di, ci);
    }

    return kg;
}

/* ──────────────────────────────────────────────
   Graph Traversal: BFS to find multi-hop paths

   Finds all paths from start_entity to target_entity
   up to max_hops steps deep through relation edges.
   ────────────────────────────────────────────── */
PathList* kg_find_paths(const KnowledgeGraph *kg, size_t start_entity,
                         size_t target_entity, size_t max_hops) {
    PathList *pl = pathlist_create();
    if (!kg || start_entity >= kg->num_entities ||
        target_entity >= kg->num_entities || max_hops == 0) {
        return pl;
    }

    /* Simple BFS path finding */
    typedef struct { size_t *ents; size_t *rels; size_t len; } BfsState;
    size_t qcap = 256;
    BfsState *queue = safe_alloc(qcap * sizeof(BfsState));
    size_t qhead = 0, qtail = 0;

    /* Initialize with start entity */
    queue[qtail].ents = safe_alloc(1 * sizeof(size_t));
    queue[qtail].rels = NULL;
    queue[qtail].ents[0] = start_entity;
    queue[qtail].len = 1;
    qtail++;

    size_t max_paths = 16;
    while (qhead < qtail && pl->count < max_paths) {
        BfsState *state = &queue[qhead];
        size_t last_ent = state->ents[state->len - 1];

        if (last_ent == target_entity && state->len > 1) {
            /* Found a path */
            if (pl->count >= pl->capacity) {
                pl->capacity *= 2;
                pl->paths = realloc(pl->paths, pl->capacity * sizeof(GraphPath));
                if (!pl->paths) { fprintf(stderr, "PathList OOM\n"); exit(1); }
            }
            GraphPath *gp = &pl->paths[pl->count];
            gp->entity_ids   = safe_alloc(state->len * sizeof(size_t));
            memcpy(gp->entity_ids, state->ents, state->len * sizeof(size_t));
            gp->relation_ids = NULL;
            gp->length       = state->len;
            pl->count++;
        } else if (state->len < max_hops) {
            /* Expand via relations */
            for (size_t r = 0; r < kg->num_relations; r++) {
                bool match = (kg->relations[r].subject_id == last_ent)
                           || (kg->relations[r].object_id == last_ent);
                if (!match) continue;

                size_t next_ent = (kg->relations[r].subject_id == last_ent)
                                ? kg->relations[r].object_id
                                : kg->relations[r].subject_id;

                /* Check if already visited */
                bool visited = false;
                for (size_t v = 0; v < state->len; v++) {
                    if (state->ents[v] == next_ent) { visited = true; break; }
                }
                if (visited) continue;

                if (qtail >= qcap) {
                    qcap *= 2;
                    queue = realloc(queue, qcap * sizeof(BfsState));
                    if (!queue) { fprintf(stderr, "BFS queue OOM\n"); exit(1); }
                }
                queue[qtail].ents = safe_alloc((state->len + 1) * sizeof(size_t));
                memcpy(queue[qtail].ents, state->ents, state->len * sizeof(size_t));
                queue[qtail].ents[state->len] = next_ent;
                queue[qtail].rels = NULL;
                queue[qtail].len = state->len + 1;
                qtail++;
            }
        }
        free(state->ents);
        free(state->rels);
        qhead++;
    }

    /* Clean up remaining queue */
    for (size_t i = qhead; i < qtail; i++) {
        free(queue[i].ents);
        free(queue[i].rels);
    }
    free(queue);

    return pl;
}

/* ──────────────────────────────────────────────
   Graph-Enhanced Scoring

   Entity Overlap: How many entities from the query
   appear in the given chunk's entities?

   Formula: overlap_score = |Q ∩ C| / |Q|

   Where Q = query entities, C = chunk entities
   ────────────────────────────────────────────── */
float kg_entity_overlap_score(const KnowledgeGraph *kg,
                               const size_t *query_entities,
                               size_t num_query_entities,
                               size_t chunk_idx) {
    if (!kg || !query_entities || num_query_entities == 0) return 0.0f;

    /* Collect entities in the target chunk */
    size_t chunk_ents[128];
    size_t chunk_count = 0;
    for (size_t i = 0; i < kg->num_entities && chunk_count < 128; i++) {
        if (kg->entities[i].chunk_index == chunk_idx) {
            chunk_ents[chunk_count++] = i;
        }
    }

    /* Count overlap */
    size_t overlap = 0;
    for (size_t qi = 0; qi < num_query_entities; qi++) {
        for (size_t ci = 0; ci < chunk_count; ci++) {
            if (query_entities[qi] == chunk_ents[ci]) {
                overlap++;
                break;
            }
        }
    }

    return (float)overlap / (float)num_query_entities;
}

/*
 * Graph Distance Score
 *
 * Measures how close the chunk's entities are to
 * the query entities in the knowledge graph.
 * Uses BFS to compute shortest path distances.
 *
 * Formula: score = max_{q in Q, c in C} 1 / (1 + dist(q, c))
 *
 * Closer entities contribute higher scores.
 */
float kg_graph_distance_score(const KnowledgeGraph *kg,
                               const size_t *query_entities,
                               size_t num_query_entities,
                               size_t chunk_idx) {
    if (!kg || !query_entities || num_query_entities == 0) return 0.0f;

    /* Collect entities in the target chunk */
    size_t chunk_ents[128];
    size_t chunk_count = 0;
    for (size_t i = 0; i < kg->num_entities && chunk_count < 128; i++) {
        if (kg->entities[i].chunk_index == chunk_idx) {
            chunk_ents[chunk_count++] = i;
        }
    }
    if (chunk_count == 0) return 0.0f;

    /* BFS distance from each query entity */
    float best_score = 0.0f;
    for (size_t qi = 0; qi < num_query_entities; qi++) {
        size_t start = query_entities[qi];
        if (start >= kg->num_entities) continue;

        /* Simple BFS to compute distances */
        size_t *dist = safe_alloc(kg->num_entities * sizeof(size_t));
        for (size_t i = 0; i < kg->num_entities; i++) dist[i] = (size_t)-1;
        dist[start] = 0;

        size_t *q = safe_alloc(kg->num_entities * sizeof(size_t));
        size_t qh = 0, qt = 0;
        q[qt++] = start;

        while (qh < qt) {
            size_t u = q[qh++];
            for (size_t r = 0; r < kg->num_relations; r++) {
                size_t v = (size_t)-1;
                if (kg->relations[r].subject_id == u)
                    v = kg->relations[r].object_id;
                else if (kg->relations[r].object_id == u)
                    v = kg->relations[r].subject_id;
                if (v < kg->num_entities && dist[v] == (size_t)-1) {
                    dist[v] = dist[u] + 1;
                    q[qt++] = v;
                }
            }
        }

        for (size_t ci = 0; ci < chunk_count; ci++) {
            size_t d = dist[chunk_ents[ci]];
            if (d != (size_t)-1) {
                float s = 1.0f / (1.0f + (float)d);
                if (s > best_score) best_score = s;
            }
        }

        free(dist);
        free(q);
    }

    return best_score;
}

/* ──────────────────────────────────────────────
   Subgraph Extraction

   Extracts entities and relations within `radius` hops
   from seed entities.
   ────────────────────────────────────────────── */
KnowledgeGraph* kg_extract_subgraph(const KnowledgeGraph *kg,
                                     const size_t *seed_entities,
                                     size_t num_seeds,
                                     size_t radius) {
    KnowledgeGraph *sub = kg_create();
    if (!kg || !seed_entities || num_seeds == 0 || radius == 0) return sub;

    /* BFS from all seeds simultaneously */
    size_t *dist = safe_alloc(kg->num_entities * sizeof(size_t));
    for (size_t i = 0; i < kg->num_entities; i++) dist[i] = (size_t)-1;

    size_t *q = safe_alloc(kg->num_entities * sizeof(size_t));
    size_t qh = 0, qt = 0;

    for (size_t s = 0; s < num_seeds; s++) {
        if (seed_entities[s] < kg->num_entities) {
            dist[seed_entities[s]] = 0;
            q[qt++] = seed_entities[s];
        }
    }

    while (qh < qt) {
        size_t u = q[qh++];
        if (dist[u] >= radius) continue;
        for (size_t r = 0; r < kg->num_relations; r++) {
            size_t v = (size_t)-1;
            if (kg->relations[r].subject_id == u)
                v = kg->relations[r].object_id;
            else if (kg->relations[r].object_id == u)
                v = kg->relations[r].subject_id;
            if (v < kg->num_entities && dist[v] == (size_t)-1) {
                dist[v] = dist[u] + 1;
                q[qt++] = v;
            }
        }
    }

    /* Copy entities within radius */
    size_t *old_to_new = safe_alloc(kg->num_entities * sizeof(size_t));
    for (size_t i = 0; i < kg->num_entities; i++) old_to_new[i] = (size_t)-1;

    for (size_t i = 0; i < kg->num_entities; i++) {
        if (dist[i] <= radius) {
            Entity *src = &kg->entities[i];
            /* Manual entity copy */
            if (sub->num_entities >= sub->entity_capacity) {
                sub->entity_capacity *= 2;
                sub->entities = realloc(sub->entities,
                                         sub->entity_capacity * sizeof(Entity));
                if (!sub->entities) { fprintf(stderr, "Subgraph OOM\n"); exit(1); }
            }
            Entity *dst = &sub->entities[sub->num_entities];
            dst->id              = sub->num_entities;
            dst->name            = safe_strdup(src->name);
            dst->normalized_name = safe_strdup(src->normalized_name);
            dst->type            = src->type;
            dst->first_char      = src->first_char;
            dst->last_char       = src->last_char;
            dst->doc_index       = src->doc_index;
            dst->chunk_index     = src->chunk_index;
            dst->confidence      = src->confidence;
            old_to_new[i] = sub->num_entities;
            sub->num_entities++;
        }
    }

    /* Copy relations where both ends are in subgraph */
    for (size_t r = 0; r < kg->num_relations; r++) {
        size_t so = old_to_new[kg->relations[r].subject_id];
        size_t oo = old_to_new[kg->relations[r].object_id];
        if (so != (size_t)-1 && oo != (size_t)-1) {
            add_relation(sub, so, oo,
                         kg->relations[r].type,
                         kg->relations[r].predicate,
                         kg->relations[r].confidence,
                         kg->relations[r].evidence_text);
        }
    }

    free(dist);
    free(q);
    free(old_to_new);
    return sub;
}

/* ──────────────────────────────────────────────
   Statistics
   ────────────────────────────────────────────── */
size_t kg_entity_count(const KnowledgeGraph *kg) {
    return kg ? kg->num_entities : 0;
}

size_t kg_relation_count(const KnowledgeGraph *kg) {
    return kg ? kg->num_relations : 0;
}

/*
 * Graph Density
 *
 * density = |E| / (|V| * (|V| - 1))
 *
 * For directed graphs. Undirected interpretation:
 * density = 2*|E| / (|V| * (|V| - 1))
 * We report the undirected density for simplicity.
 */
float kg_density(const KnowledgeGraph *kg) {
    if (!kg || kg->num_entities < 2) return 0.0f;
    size_t V = kg->num_entities;
    double max_edges = (double)V * (double)(V - 1);
    return (float)(2.0 * (double)kg->num_relations / max_edges);
}