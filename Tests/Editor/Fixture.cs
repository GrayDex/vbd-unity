using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Unity.Mathematics;
using UnityEngine;

namespace Vbd.Tests.Editor
{
    // A Newton reference run written by Tools~/newton/fixture.py.
    sealed class Fixture
    {
        [Serializable] class Param { public string name; public float value; }
        [Serializable] class Entry { public string name; public string dtype; public int[] shape; public int offset; }
        [Serializable] class Header { public string scene; public Param[] @params; public Entry[] arrays; }

        public static string Dir => Path.GetFullPath("Packages/io.github.graydex.vbd/Tests/Fixtures~");

        readonly Header _header;
        readonly byte[] _data;

        public Fixture(string scene)
        {
            var bytes = File.ReadAllBytes(Path.Combine(Dir, scene + ".vbdf"));
            if (Encoding.ASCII.GetString(bytes, 0, 4) != "VBDF") throw new InvalidDataException(scene);
            var headerLength = BitConverter.ToInt32(bytes, 4);
            _header = JsonUtility.FromJson<Header>(Encoding.UTF8.GetString(bytes, 8, headerLength));
            _data = new byte[bytes.Length - 8 - headerLength];
            Buffer.BlockCopy(bytes, 8 + headerLength, _data, 0, _data.Length);
        }

        public float Value(string name) => _header.@params.First(p => p.name == name).value;
        public int[] Shape(string name) => Find(name).shape;

        public T[] Array<T>(string name) where T : unmanaged
        {
            var e = Find(name);
            var bytes = e.shape.Aggregate(4, (a, b) => a * b);
            return MemoryMarshal.Cast<byte, T>(new ReadOnlySpan<byte>(_data, e.offset, bytes)).ToArray();
        }

        // Newton matrices are row-major; Unity.Mathematics stores columns.
        public float3x3[] Matrices(string name) => Array<float3x3>(name).Select(math.transpose).ToArray();

        Entry Find(string name) => _header.arrays.FirstOrDefault(a => a.name == name)
                                   ?? throw new ArgumentException($"{_header.scene}: no array {name}");
    }
}
