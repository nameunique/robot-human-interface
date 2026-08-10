using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

public static class FbxToObj
{
    private const float RobotScale = 0.35f;

    public static void ExportAll()
    {
        try
        {
            string repositoryRoot = Path.GetFullPath(
                Path.Combine(Application.dataPath, "..", "..", ".."));
            string sourceRoot = Path.Combine(repositoryRoot, "assets", "unity_fbx");
            string inputRoot = Path.Combine(Application.dataPath, "Input");
            string outputRoot = Path.Combine(repositoryRoot, "models", "humanoid", "meshes");

            if (!Directory.Exists(sourceRoot))
            {
                throw new DirectoryNotFoundException($"FBX source directory not found: {sourceRoot}");
            }

            if (Directory.Exists(inputRoot))
            {
                Directory.Delete(inputRoot, true);
            }
            Directory.CreateDirectory(inputRoot);
            Directory.CreateDirectory(outputRoot);

            string[] sourceFiles = Directory.GetFiles(sourceRoot, "*.fbx", SearchOption.AllDirectories);
            Array.Sort(sourceFiles, StringComparer.OrdinalIgnoreCase);
            if (sourceFiles.Length != 21)
            {
                throw new InvalidOperationException(
                    $"Expected 21 FBX files, found {sourceFiles.Length} in {sourceRoot}");
            }

            foreach (string sourcePath in sourceFiles)
            {
                string relativePath = Path.GetRelativePath(sourceRoot, sourcePath);
                string destinationPath = Path.Combine(inputRoot, relativePath);
                Directory.CreateDirectory(Path.GetDirectoryName(destinationPath));
                File.Copy(sourcePath, destinationPath, true);
            }

            AssetDatabase.Refresh(ImportAssetOptions.ForceSynchronousImport);

            int totalVertices = 0;
            int totalTriangles = 0;
            foreach (string sourcePath in sourceFiles)
            {
                string relativePath = Path.GetRelativePath(sourceRoot, sourcePath).Replace('\\', '/');
                string assetPath = "Assets/Input/" + relativePath;
                ModelImporter importer = AssetImporter.GetAtPath(assetPath) as ModelImporter;
                if (importer == null)
                {
                    throw new InvalidOperationException($"Unity did not create a ModelImporter for {assetPath}");
                }

                importer.isReadable = true;
                importer.importAnimation = false;
                importer.importCameras = false;
                importer.importLights = false;
                importer.SaveAndReimport();

                GameObject modelAsset = AssetDatabase.LoadAssetAtPath<GameObject>(assetPath);
                if (modelAsset == null)
                {
                    throw new InvalidOperationException($"Cannot load imported FBX: {assetPath}");
                }

                GameObject instance = UnityEngine.Object.Instantiate(modelAsset);
                try
                {
                    instance.transform.SetPositionAndRotation(Vector3.zero, Quaternion.identity);
                    instance.transform.localScale = Vector3.one;
                    MeshFilter[] filters = instance.GetComponentsInChildren<MeshFilter>(true);
                    if (filters.Length == 0)
                    {
                        throw new InvalidOperationException($"No MeshFilter found in {assetPath}");
                    }

                    string outputName = Path.GetFileNameWithoutExtension(sourcePath) + ".obj";
                    string outputPath = Path.Combine(outputRoot, outputName);
                    (int vertices, int triangles) = WriteObj(outputPath, outputName, filters);
                    totalVertices += vertices;
                    totalTriangles += triangles;
                    Debug.Log($"Exported {outputName}: {vertices} vertices, {triangles} triangles");
                }
                finally
                {
                    UnityEngine.Object.DestroyImmediate(instance);
                }
            }

            Debug.Log(
                $"FBX to OBJ export complete: {sourceFiles.Length} files, " +
                $"{totalVertices} vertices, {totalTriangles} triangles -> {outputRoot}");
            EditorApplication.Exit(0);
        }
        catch (Exception exception)
        {
            Debug.LogException(exception);
            EditorApplication.Exit(1);
        }
    }

    private static (int vertices, int triangles) WriteObj(
        string outputPath,
        string objectName,
        IReadOnlyList<MeshFilter> filters)
    {
        CultureInfo invariant = CultureInfo.InvariantCulture;
        var builder = new StringBuilder(1024 * 1024);
        builder.Append("# Generated from the Unity FBX import by robot-human-interface\n");
        builder.Append("o ").Append(objectName).Append('\n');

        int vertexOffset = 1;
        int vertexCount = 0;
        int triangleCount = 0;
        var transformedPositions = new List<Vector3[]>();
        var transformedNormals = new List<Vector3[]>();

        foreach (MeshFilter filter in filters)
        {
            Mesh mesh = filter.sharedMesh;
            if (mesh == null)
            {
                continue;
            }

            Matrix4x4 localToWorld = filter.transform.localToWorldMatrix;
            Matrix4x4 normalMatrix = localToWorld.inverse.transpose;
            Vector3[] vertices = mesh.vertices;
            Vector3[] sourceNormals = mesh.normals;
            Vector3[] positions = new Vector3[vertices.Length];
            Vector3[] normals = new Vector3[vertices.Length];

            for (int index = 0; index < vertices.Length; index++)
            {
                Vector3 unityPosition = localToWorld.MultiplyPoint3x4(vertices[index]);
                Vector3 mujocoPosition = ToMujocoPosition(unityPosition);
                positions[index] = mujocoPosition;
                builder.Append("v ")
                    .Append(mujocoPosition.x.ToString("R", invariant)).Append(' ')
                    .Append(mujocoPosition.y.ToString("R", invariant)).Append(' ')
                    .Append(mujocoPosition.z.ToString("R", invariant)).Append('\n');
            }

            bool hasNormals = sourceNormals.Length == vertices.Length;
            for (int index = 0; index < vertices.Length; index++)
            {
                Vector3 unityNormal = hasNormals
                    ? normalMatrix.MultiplyVector(sourceNormals[index]).normalized
                    : Vector3.up;
                Vector3 mujocoNormal = ToMujocoDirection(unityNormal).normalized;
                normals[index] = mujocoNormal;
                builder.Append("vn ")
                    .Append(mujocoNormal.x.ToString("R", invariant)).Append(' ')
                    .Append(mujocoNormal.y.ToString("R", invariant)).Append(' ')
                    .Append(mujocoNormal.z.ToString("R", invariant)).Append('\n');
            }

            transformedPositions.Add(positions);
            transformedNormals.Add(normals);
            vertexCount += vertices.Length;
        }

        int filterIndex = 0;
        foreach (MeshFilter filter in filters)
        {
            Mesh mesh = filter.sharedMesh;
            if (mesh == null)
            {
                continue;
            }

            Vector3[] positions = transformedPositions[filterIndex];
            Vector3[] normals = transformedNormals[filterIndex];
            filterIndex++;
            int[] indices = mesh.triangles;
            if (indices.Length % 3 != 0)
            {
                throw new InvalidOperationException($"Non-triangular index buffer in {mesh.name}");
            }

            for (int index = 0; index < indices.Length; index += 3)
            {
                int a = indices[index];
                int b = indices[index + 1];
                int c = indices[index + 2];
                Vector3 faceNormal = Vector3.Cross(positions[b] - positions[a], positions[c] - positions[a]);
                Vector3 importedNormal = normals[a] + normals[b] + normals[c];
                if (Vector3.Dot(faceNormal, importedNormal) < 0.0f)
                {
                    (b, c) = (c, b);
                }

                WriteFaceVertex(builder, vertexOffset + a);
                WriteFaceVertex(builder, vertexOffset + b);
                WriteFaceVertex(builder, vertexOffset + c, true);
                triangleCount++;
            }

            vertexOffset += mesh.vertexCount;
        }

        File.WriteAllText(outputPath, builder.ToString(), new UTF8Encoding(false));
        return (vertexCount, triangleCount);
    }

    private static void WriteFaceVertex(StringBuilder builder, int index, bool end = false)
    {
        if (builder.Length > 0 && builder[builder.Length - 1] == '\n')
        {
            builder.Append("f ");
        }
        else
        {
            builder.Append(' ');
        }
        builder.Append(index).Append("//").Append(index);
        if (end)
        {
            builder.Append('\n');
        }
    }

    private static Vector3 ToMujocoPosition(Vector3 unityPosition)
    {
        return new Vector3(
            -unityPosition.z * RobotScale - 0.035f,
            -unityPosition.x * RobotScale,
            unityPosition.y * RobotScale);
    }

    private static Vector3 ToMujocoDirection(Vector3 unityDirection)
    {
        return new Vector3(-unityDirection.z, -unityDirection.x, unityDirection.y);
    }
}
