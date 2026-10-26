using System;
using System.Collections.Generic;
using Unity.Mathematics;

namespace Vbd
{
    public enum ColoringAlgorithm
    {
        Mcs = 0,
        Greedy = 1
    }

    // Port of newton/_src/sim/graph_coloring.py and NVIDIA Warp's warp/native/coloring.cpp.
    // Neighbor order, bucket order and the sort are kept identical so colors match Warp's bit for bit.
    public static class Coloring
    {
        sealed class Graph
        {
            public readonly int NumNodes;
            public readonly int[] Flatten;
            public readonly int[] Offsets;
            public readonly int[] NodeColors;

            // wp::Graph::Graph: adjacency in edge-list order.
            public Graph(int numNodes, int2[] edges)
            {
                NumNodes = numNodes;
                Offsets = new int[numNodes + 1];
                NodeColors = new int[numNodes];
                Array.Fill(NodeColors, -1);
                var degrees = new int[numNodes];
                foreach (var e in edges)
                {
                    if (e.x < 0 || e.x >= numNodes || e.y < 0 || e.y >= numNodes)
                        throw new ArgumentException($"Graph edge ({e.x}, {e.y}) is outside [0, {numNodes})");
                    degrees[e.x]++;
                    degrees[e.y]++;
                }
                for (var i = 0; i < numNodes; i++) Offsets[i + 1] = Offsets[i] + degrees[i];
                Flatten = new int[Offsets[numNodes]];
                var fill = new int[numNodes];
                foreach (var e in edges)
                {
                    Flatten[Offsets[e.x] + fill[e.x]++] = e.y;
                    Flatten[Offsets[e.y] + fill[e.y]++] = e.x;
                }
            }

            public int Degree(int node) => Offsets[node + 1] - Offsets[node];
            public int Neighbor(int node, int i) => Flatten[Offsets[node] + i];
        }

        // construct_tetmesh_graph_edges: unique sorted edges (a < b) of the active tetrahedra.
        public static int2[] ConstructTetmeshGraphEdges(IReadOnlyList<int4> tets, IReadOnlyList<bool> active = null)
        {
            var keys = new List<long>(tets.Count * 6);
            for (var t = 0; t < tets.Count; t++)
            {
                if (active != null && !active[t]) continue;
                var v = tets[t];
                for (var a = 0; a < 3; a++)
                for (var b = a + 1; b < 4; b++)
                {
                    int lo = math.min(v[a], v[b]), hi = math.max(v[a], v[b]);
                    keys.Add(((long)lo << 32) | (uint)hi);
                }
            }
            keys.Sort();
            var edges = new List<int2>(keys.Count);
            for (var i = 0; i < keys.Count; i++)
                if (i == 0 || keys[i] != keys[i - 1])
                    edges.Add(new int2((int)(keys[i] >> 32), (int)(keys[i] & 0xffffffff)));
            return edges.ToArray();
        }

        // graph_coloring_assign + optional graph_coloring_balance + convert_to_color_groups.
        public static List<int[]> ColorGraph(int numNodes, int2[] edges, bool balanceColors = true,
            float targetMaxMinColorRatio = 1.1f, ColoringAlgorithm algorithm = ColoringAlgorithm.Mcs)
        {
            if (numNodes == 0) return new List<int[]>();
            var colors = new int[numNodes];
            var numColors = Assign(numNodes, edges, algorithm, colors);
            if (balanceColors) Balance(numNodes, edges, colors, numColors, targetMaxMinColorRatio);
            return ToGroups(numColors, colors);
        }

        // color_rigid_bodies: bodies adjacent through a joint (or an additional edge) get different colors.
        public static List<int[]> ColorRigidBodies(int numBodies, IReadOnlyList<int> jointParent,
            IReadOnlyList<int> jointChild, bool balanceColors = true, float targetMaxMinColorRatio = 1.1f,
            ColoringAlgorithm algorithm = ColoringAlgorithm.Mcs, IReadOnlyList<int2> additionalEdges = null)
        {
            if (numBodies == 0) return new List<int[]>();
            var parents = jointParent ?? Array.Empty<int>();
            var children = jointChild ?? Array.Empty<int>();
            if (parents.Count != children.Count)
                throw new ArgumentException("jointParent and jointChild must have the same length");
            var edges = new List<int2>();
            for (var i = 0; i < parents.Count; i++)
                if (parents[i] != -1 && children[i] != -1 && parents[i] != children[i])
                    edges.Add(new int2(parents[i], children[i]));
            if (additionalEdges != null)
                foreach (var e in additionalEdges)
                    if (e.x != -1 && e.y != -1 && e.x != e.y)
                        edges.Add(e);
            if (edges.Count == 0)
            {
                var all = new int[numBodies];
                for (var i = 0; i < numBodies; i++) all[i] = i;
                return new List<int[]> { all };
            }
            return ColorGraph(numBodies, edges.ToArray(), balanceColors, targetMaxMinColorRatio, algorithm);
        }

        // validate_graph_coloring
        public static bool IsValid(int2[] edges, int[] colors)
        {
            foreach (var e in edges)
                if (colors[e.x] == colors[e.y])
                    return false;
            return true;
        }

        // wp_graph_coloring
        public static int Assign(int numNodes, int2[] edges, ColoringAlgorithm algorithm, int[] nodeColors)
        {
            var graph = new Graph(numNodes, edges);
            var count = algorithm == ColoringAlgorithm.Mcs ? McsVector(graph) : DegreeOrderedGreedy(graph);
            Array.Copy(graph.NodeColors, nodeColors, numNodes);
            return count;
        }

        // wp_balance_coloring; maxIterations < 0 means numNodes, as warp.utils.graph_coloring_balance.
        public static float Balance(int numNodes, int2[] edges, int[] nodeColors, int numColors,
            float targetMaxMinRatio, int maxIterations = -1)
        {
            if (maxIterations < 0) maxIterations = numNodes;
            var graph = new Graph(numNodes, edges);
            Array.Copy(nodeColors, graph.NodeColors, numNodes);
            if (numColors <= 1) return 1f;
            var groups = ToGroupLists(numColors, graph.NodeColors);
            var ratio = BalanceColorGroups(targetMaxMinRatio, maxIterations, graph, groups);
            Array.Copy(graph.NodeColors, nodeColors, numNodes);
            return ratio;
        }

        public static List<int[]> ToGroups(int numColors, int[] nodeColors)
        {
            var lists = ToGroupLists(numColors, nodeColors);
            var groups = new List<int[]>(numColors);
            foreach (var l in lists) groups.Add(l.ToArray());
            return groups;
        }

        static List<int>[] ToGroupLists(int numColors, int[] nodeColors)
        {
            var groups = new List<int>[numColors];
            for (var c = 0; c < numColors; c++) groups[c] = new List<int>();
            for (var i = 0; i < nodeColors.Length; i++)
            {
                var c = nodeColors[i];
                if (c < 0 || c >= numColors) throw new ArgumentException($"Node {i} has color {c}, valid are [0, {numColors})");
                groups[c].Add(i);
            }
            return groups;
        }

        static float FindLargestSmallestGroups(List<int>[] groups, out int biggest, out int smallest)
        {
            if (groups.Length == 0)
            {
                biggest = smallest = -1;
                return 1f;
            }
            biggest = smallest = 0;
            var max = groups[0].Count;
            var min = groups[0].Count;
            for (var c = 0; c < groups.Length; c++)
            {
                if (max < groups[c].Count)
                {
                    biggest = c;
                    max = groups[c].Count;
                }
                if (min > groups[c].Count)
                {
                    smallest = c;
                    min = groups[c].Count;
                }
            }
            return (float)groups[biggest].Count / groups[smallest].Count;
        }

        static bool ColorChangeable(Graph graph, int node, int targetColor)
        {
            for (var i = 0; i < graph.Degree(node); i++)
                if (graph.NodeColors[graph.Neighbor(node, i)] == targetColor)
                    return false;
            return true;
        }

        static int FindChangeableNodeInCategory(Graph graph, List<int>[] groups, int source, int target)
        {
            var group = groups[source];
            for (var i = 0; i < group.Count; i++)
                if (ColorChangeable(graph, group[i], target))
                    return i;
            return -1;
        }

        static void ChangeColor(int color, int indexInGroup, int target, int[] nodeColors, List<int>[] groups)
        {
            var node = groups[color][indexInGroup];
            nodeColors[node] = target;
            var g = groups[color];
            (g[indexInGroup], g[^1]) = (g[^1], g[indexInGroup]);
            g.RemoveAt(g.Count - 1);
            groups[target].Add(node);
        }

        // balance_color_groups
        static float BalanceColorGroups(float targetRatio, int maxIterations, Graph graph, List<int>[] groups)
        {
            var ratio = -1f;
            var lastNode = -1;
            var lastColor = -1;
            var iteration = 0;
            while (true)
            {
                var prev = ratio;
                ratio = FindLargestSmallestGroups(groups, out var biggest, out var smallest);
                if (prev > 0f && prev < ratio)
                {
                    graph.NodeColors[lastNode] = lastColor;
                    return prev;
                }
                if (groups[biggest].Count - groups[smallest].Count <= 2 || ratio <= targetRatio) return ratio;

                var sourceGroup = biggest;
                var index = FindChangeableNodeInCategory(graph, groups, biggest, smallest);
                if (index == -1)
                    for (var c = 0; c < groups.Length; c++)
                    {
                        if (c == biggest || c == smallest) continue;
                        index = FindChangeableNodeInCategory(graph, groups, c, smallest);
                        if (index == -1) continue;
                        sourceGroup = c;
                        break;
                    }
                if (index == -1 || iteration == maxIterations) return ratio;

                lastNode = groups[sourceGroup][index];
                lastColor = sourceGroup;
                ChangeColor(sourceGroup, index, smallest, graph.NodeColors, groups);
                iteration++;
            }
        }

        // graph_coloring_ordered_greedy
        static int OrderedGreedy(int[] order, Graph graph)
        {
            var maxColor = -1;
            var used = new List<bool>(128);
            foreach (var node in order)
            {
                if (maxColor == -1)
                {
                    maxColor++;
                    graph.NodeColors[node] = maxColor;
                    continue;
                }
                used.Clear();
                for (var c = 0; c <= maxColor; c++) used.Add(false);
                for (var i = 0; i < graph.Degree(node); i++)
                {
                    var nc = graph.NodeColors[graph.Neighbor(node, i)];
                    if (nc >= 0) used[nc] = true;
                }
                var usable = used.IndexOf(false);
                if (usable == -1) graph.NodeColors[node] = ++maxColor;
                else graph.NodeColors[node] = usable;
            }
            return maxColor + 1;
        }

        // graph_coloring_degree_ordered_greedy: std::sort by descending degree (libstdc++ introsort, not stable).
        static int DegreeOrderedGreedy(Graph graph)
        {
            var degrees = new int[graph.NumNodes];
            var order = new int[graph.NumNodes];
            for (var i = 0; i < degrees.Length; i++)
            {
                degrees[i] = graph.Degree(i);
                order[i] = i;
            }
            StdSort.Sort(order, (l, r) => degrees[l] > degrees[r]);
            return OrderedGreedy(order, graph);
        }

        // NodeWeightBuckets
        sealed class NodeWeightBuckets
        {
            List<int>[] _buckets;
            readonly int[] _indexInBucket;
            readonly int[] _weights;
            int _maxWeight;

            public NodeWeightBuckets(int numNodes)
            {
                _weights = new int[numNodes];
                _indexInBucket = new int[numNodes];
                Array.Fill(_indexInBucket, -1);
                _buckets = new List<int>[64];
                for (var i = 0; i < _buckets.Length; i++) _buckets[i] = new List<int>();
            }

            public int Weight(int node) => _weights[node];

            public void Add(int weight, int node)
            {
                if (weight >= _buckets.Length)
                {
                    var old = _buckets.Length;
                    Array.Resize(ref _buckets, weight + 1);
                    for (var i = old; i < _buckets.Length; i++) _buckets[i] = new List<int>();
                }
                _indexInBucket[node] = _buckets[weight].Count;
                _weights[node] = weight;
                _buckets[weight].Add(node);
                if (_maxWeight < weight) _maxWeight = weight;
            }

            // Takes the first element of the top bucket (breadth-first like) and moves the last one into its place.
            public int PopMax()
            {
                var bucket = _buckets[_maxWeight];
                var node = bucket[0];
                _indexInBucket[node] = -1;
                if (bucket.Count > 1)
                {
                    _indexInBucket[bucket[^1]] = 0;
                    bucket[0] = bucket[^1];
                }
                bucket.RemoveAt(bucket.Count - 1);
                _weights[node] = -1;
                if (bucket.Count == 0)
                {
                    var newMax = 0;
                    for (var b = _maxWeight - 1; b >= 0; b--)
                        if (_buckets[b].Count > 0)
                        {
                            newMax = b;
                            break;
                        }
                    _maxWeight = newMax;
                }
                return node;
            }

            public void Increase(int node)
            {
                var weight = _weights[node];
                var bucket = _buckets[weight];
                var index = _indexInBucket[node];
                _indexInBucket[bucket[^1]] = index;
                bucket[index] = bucket[^1];
                bucket.RemoveAt(bucket.Count - 1);
                Add(weight + 1, node);
            }

            public bool Empty => _maxWeight <= 0 && _buckets[0].Count == 0;
        }

        // graph_coloring_mcs_vector: maximum cardinality search order, then greedy.
        static int McsVector(Graph graph)
        {
            var ordering = new int[graph.NumNodes];
            var buckets = new NodeWeightBuckets(graph.NumNodes);
            buckets.Add(0, 0);
            for (var i = 0; i < graph.NumNodes; i++)
            {
                if (buckets.Empty)
                {
                    var next = -1;
                    for (var n = 0; n < graph.NumNodes; n++)
                        if (buckets.Weight(n) >= 0)
                        {
                            next = n;
                            break;
                        }
                    buckets.Add(0, next);
                }
                var max = buckets.PopMax();
                ordering[i] = max;
                for (var j = 0; j < graph.Degree(max); j++)
                {
                    var neighbor = graph.Neighbor(max, j);
                    var w = buckets.Weight(neighbor);
                    if (w == 0) buckets.Add(1, neighbor);
                    else if (w > 0) buckets.Increase(neighbor);
                }
            }
            return OrderedGreedy(ordering, graph);
        }

        // libstdc++ std::sort (introsort with insertion sort finish), so ties order as in Warp built with GCC.
        static class StdSort
        {
            const int Threshold = 16;

            public static void Sort(int[] a, Func<int, int, bool> less)
            {
                var n = a.Length;
                if (n < 2) return;
                IntroLoop(a, 0, n, 2 * (31 - (int)math.lzcnt((uint)n)), less);
                FinalInsertionSort(a, 0, n, less);
            }

            static void IntroLoop(int[] a, int first, int last, int depth, Func<int, int, bool> less)
            {
                while (last - first > Threshold)
                {
                    if (depth == 0)
                    {
                        MakeHeap(a, first, last, less);
                        SortHeap(a, first, last, less);
                        return;
                    }
                    depth--;
                    var mid = first + (last - first) / 2;
                    MoveMedianToFirst(a, first, first + 1, mid, last - 1, less);
                    var cut = UnguardedPartition(a, first + 1, last, first, less);
                    IntroLoop(a, cut, last, depth, less);
                    last = cut;
                }
            }

            static void MoveMedianToFirst(int[] a, int result, int x, int y, int z, Func<int, int, bool> less)
            {
                if (less(a[x], a[y]))
                {
                    if (less(a[y], a[z])) Swap(a, result, y);
                    else if (less(a[x], a[z])) Swap(a, result, z);
                    else Swap(a, result, x);
                }
                else if (less(a[x], a[z])) Swap(a, result, x);
                else if (less(a[y], a[z])) Swap(a, result, z);
                else Swap(a, result, y);
            }

            static int UnguardedPartition(int[] a, int first, int last, int pivot, Func<int, int, bool> less)
            {
                while (true)
                {
                    while (less(a[first], a[pivot])) first++;
                    last--;
                    while (less(a[pivot], a[last])) last--;
                    if (!(first < last)) return first;
                    Swap(a, first, last);
                    first++;
                }
            }

            static void FinalInsertionSort(int[] a, int first, int last, Func<int, int, bool> less)
            {
                if (last - first > Threshold)
                {
                    InsertionSort(a, first, first + Threshold, less);
                    for (var i = first + Threshold; i < last; i++) UnguardedLinearInsert(a, i, less);
                }
                else InsertionSort(a, first, last, less);
            }

            static void InsertionSort(int[] a, int first, int last, Func<int, int, bool> less)
            {
                for (var i = first + 1; i < last; i++)
                {
                    if (less(a[i], a[first]))
                    {
                        var value = a[i];
                        Array.Copy(a, first, a, first + 1, i - first);
                        a[first] = value;
                    }
                    else UnguardedLinearInsert(a, i, less);
                }
            }

            static void UnguardedLinearInsert(int[] a, int last, Func<int, int, bool> less)
            {
                var value = a[last];
                var next = last - 1;
                while (less(value, a[next]))
                {
                    a[last] = a[next];
                    last = next;
                    next--;
                }
                a[last] = value;
            }

            static void MakeHeap(int[] a, int first, int last, Func<int, int, bool> less)
            {
                var len = last - first;
                if (len < 2) return;
                var parent = (len - 2) / 2;
                while (true)
                {
                    AdjustHeap(a, first, parent, len, a[first + parent], less);
                    if (parent == 0) return;
                    parent--;
                }
            }

            static void SortHeap(int[] a, int first, int last, Func<int, int, bool> less)
            {
                while (last - first > 1)
                {
                    last--;
                    var value = a[last];
                    a[last] = a[first];
                    AdjustHeap(a, first, 0, last - first, value, less);
                }
            }

            static void AdjustHeap(int[] a, int first, int hole, int len, int value, Func<int, int, bool> less)
            {
                var top = hole;
                var child = hole;
                while (child < (len - 1) / 2)
                {
                    child = 2 * (child + 1);
                    if (less(a[first + child], a[first + child - 1])) child--;
                    a[first + hole] = a[first + child];
                    hole = child;
                }
                if ((len & 1) == 0 && child == (len - 2) / 2)
                {
                    child = 2 * (child + 1);
                    a[first + hole] = a[first + child - 1];
                    hole = child - 1;
                }
                var parent = (hole - 1) / 2;
                while (hole > top && less(a[first + parent], value))
                {
                    a[first + hole] = a[first + parent];
                    hole = parent;
                    parent = (hole - 1) / 2;
                }
                a[first + hole] = value;
            }

            static void Swap(int[] a, int i, int j) => (a[i], a[j]) = (a[j], a[i]);
        }
    }
}
