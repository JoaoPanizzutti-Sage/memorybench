export interface EntityNode {
  name: string
  type: string
  summary: string
  sessionIds: Set<string>
}

export interface RelationshipEdge {
  source: string
  target: string
  relation: string
  date?: string
  sessionId: string
}

export interface GraphSearchResult {
  entities: Array<{ name: string; type: string; summary: string }>
  relationships: Array<{ source: string; target: string; relation: string; date?: string }>
}

const MAX_GRAPH_ENTITIES = 10
const MAX_GRAPH_RELATIONSHIPS = 20

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")
}

export class EntityGraph {
  private nodes = new Map<string, EntityNode>()
  private edgeSet = new Set<string>()
  private adjacency = new Map<string, RelationshipEdge[]>()
  private nameIndex = new Map<string, Set<string>>()

  addEntity(name: string, type: string, summary: string, sessionId: string): void {
    const canonical = name.trim()
    if (!canonical) return

    const existing = this.nodes.get(canonical)
    if (existing) {
      existing.sessionIds.add(sessionId)
      if (summary && !existing.summary.includes(summary.substring(0, 40))) {
        existing.summary = (existing.summary + " " + summary).substring(0, 500)
      }
    } else {
      this.nodes.set(canonical, {
        name: canonical,
        type: type.toLowerCase(),
        summary,
        sessionIds: new Set([sessionId]),
      })
    }

    // Index full name and individual parts (>2 chars) -> Set of canonical names
    const key = canonical.toLowerCase()
    if (!this.nameIndex.has(key)) this.nameIndex.set(key, new Set())
    this.nameIndex.get(key)!.add(canonical)

    for (const part of canonical.split(/\s+/)) {
      if (part.length > 2) {
        const pk = part.toLowerCase()
        if (!this.nameIndex.has(pk)) this.nameIndex.set(pk, new Set())
        this.nameIndex.get(pk)!.add(canonical)
      }
    }
  }

  addRelationship(edge: RelationshipEdge): void {
    const key = `${edge.source}|${edge.relation}|${edge.target}`
    if (this.edgeSet.has(key)) return
    this.edgeSet.add(key)

    // Adjacency list for fast traversal
    if (!this.adjacency.has(edge.source)) this.adjacency.set(edge.source, [])
    this.adjacency.get(edge.source)!.push(edge)
    if (!this.adjacency.has(edge.target)) this.adjacency.set(edge.target, [])
    this.adjacency.get(edge.target)!.push(edge)
  }

  findEntitiesInQuery(query: string): string[] {
    const queryLower = query.toLowerCase()
    const matched = new Set<string>()

    for (const [key, canonicals] of this.nameIndex) {
      if (key.length <= 2) continue
      // Word-boundary match to avoid "tom" matching inside "tomato"
      const regex = new RegExp(`\\b${escapeRegex(key)}\\b`)
      if (regex.test(queryLower)) {
        for (const c of canonicals) matched.add(c)
      }
    }

    return [...matched]
  }

  getContext(entityNames: string[], maxHops = 2): GraphSearchResult {
    const visitedNodes = new Set<string>()
    const resultEntities: GraphSearchResult["entities"] = []
    const resultRels: GraphSearchResult["relationships"] = []
    const seenEdges = new Set<string>()

    // Process seed entities first (not counted as a hop)
    for (const name of entityNames) {
      if (visitedNodes.has(name)) continue
      visitedNodes.add(name)
      const node = this.nodes.get(name)
      if (node && resultEntities.length < MAX_GRAPH_ENTITIES) {
        resultEntities.push({ name: node.name, type: node.type, summary: node.summary })
      }
    }

    let frontier = [...entityNames]

    for (let hop = 0; hop < maxHops && frontier.length > 0; hop++) {
      const nextFrontier: string[] = []

      for (const name of frontier) {
        const edges = this.adjacency.get(name) || []
        for (const edge of edges) {
          const edgeKey = `${edge.source}|${edge.relation}|${edge.target}`
          if (seenEdges.has(edgeKey)) continue
          seenEdges.add(edgeKey)

          if (resultRels.length < MAX_GRAPH_RELATIONSHIPS) {
            resultRels.push({
              source: edge.source,
              target: edge.target,
              relation: edge.relation,
              date: edge.date,
            })
          }

          const other = edge.source === name ? edge.target : edge.source
          if (!visitedNodes.has(other)) {
            visitedNodes.add(other)
            nextFrontier.push(other)

            const otherNode = this.nodes.get(other)
            if (otherNode && resultEntities.length < MAX_GRAPH_ENTITIES) {
              resultEntities.push({ name: otherNode.name, type: otherNode.type, summary: otherNode.summary })
            }
          }
        }
      }

      frontier = nextFrontier
    }

    return { entities: resultEntities, relationships: resultRels }
  }

  get nodeCount(): number {
    return this.nodes.size
  }

  get edgeCount(): number {
    return this.edgeSet.size
  }

  save(): SerializedGraph {
    const nodes: SerializedEntityNode[] = []
    for (const node of this.nodes.values()) {
      nodes.push({ name: node.name, type: node.type, summary: node.summary, sessionIds: [...node.sessionIds] })
    }
    const edges: RelationshipEdge[] = []
    const seen = new Set<string>()
    for (const edgeList of this.adjacency.values()) {
      for (const edge of edgeList) {
        const key = `${edge.source}|${edge.relation}|${edge.target}`
        if (!seen.has(key)) {
          seen.add(key)
          edges.push(edge)
        }
      }
    }
    return { nodes, edges }
  }

  load(data: SerializedGraph): void {
    this.clear()
    for (const node of data.nodes) {
      this.nodes.set(node.name, {
        name: node.name,
        type: node.type,
        summary: node.summary,
        sessionIds: new Set(node.sessionIds),
      })
      // Rebuild name index
      const key = node.name.toLowerCase()
      if (!this.nameIndex.has(key)) this.nameIndex.set(key, new Set())
      this.nameIndex.get(key)!.add(node.name)
      for (const part of node.name.split(/\s+/)) {
        if (part.length > 2) {
          const pk = part.toLowerCase()
          if (!this.nameIndex.has(pk)) this.nameIndex.set(pk, new Set())
          this.nameIndex.get(pk)!.add(node.name)
        }
      }
    }
    for (const edge of data.edges) {
      this.addRelationship(edge)
    }
  }

  clear(): void {
    this.nodes.clear()
    this.edgeSet.clear()
    this.adjacency.clear()
    this.nameIndex.clear()
  }
}

interface SerializedEntityNode {
  name: string
  type: string
  summary: string
  sessionIds: string[]
}

export interface SerializedGraph {
  nodes: SerializedEntityNode[]
  edges: RelationshipEdge[]
}
