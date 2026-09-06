package com.iykyk.collage

import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import java.io.File
import java.io.FileOutputStream

class MainActivity : ComponentActivity() {

    private val viewModel: ProcessingViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    ProcessingScreen(viewModel)
                }
            }
        }
    }
}

@Composable
fun ProcessingScreen(viewModel: ProcessingViewModel) {
    val context = LocalContext.current
    val state by viewModel.state.collectAsState()

    val pickVideoLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let { viewModel.processVideo(it) }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        when (val s = state) {
            is ProcessingState.Idle -> {
                Text("Pick a portrait video to build a collage", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(16.dp))
                Button(onClick = { pickVideoLauncher.launch("video/*") }) {
                    Text("Choose video")
                }
            }

            is ProcessingState.ExtractingFrames -> {
                Text("Reading video…")
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(progress = { s.progress }, modifier = Modifier.fillMaxWidth())
            }

            is ProcessingState.DetectingFaces -> {
                Text("Detecting faces…")
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(progress = { s.progress }, modifier = Modifier.fillMaxWidth())
            }

            is ProcessingState.ClusteringIdentities -> {
                Text("Grouping appearances by person…")
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }

            is ProcessingState.ComposingCollage -> {
                Text("Building collage…")
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
            }

            is ProcessingState.Done -> {
                Text("${s.identities.size} people found", style = MaterialTheme.typography.titleMedium)
                Spacer(Modifier.height(12.dp))
                Image(
                    bitmap = s.collage.asImageBitmap(),
                    contentDescription = "Collage",
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(Modifier.height(16.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(onClick = { saveToGallery(context, s.collage) }) {
                        Text("Save")
                    }
                    Button(onClick = { shareBitmap(context, s.collage) }) {
                        Text("Share")
                    }
                }
                Spacer(Modifier.height(16.dp))
                TextButton(onClick = { viewModel.reset() }) {
                    Text("Process another video")
                }
            }

            is ProcessingState.Error -> {
                Text("Something went wrong: ${s.message}", color = MaterialTheme.colorScheme.error)
                Spacer(Modifier.height(16.dp))
                Button(onClick = { viewModel.reset() }) {
                    Text("Try again")
                }
            }
        }
    }
}

private fun saveToGallery(context: android.content.Context, bitmap: Bitmap) {
    val filename = "iykyk_collage_${System.currentTimeMillis()}.jpg"
    val resolver = context.contentResolver
    val values = android.content.ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, filename)
        put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/iykyk")
    }
    val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
    resolver.openOutputStream(uri)?.use { out ->
        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out)
    }
}

private fun shareBitmap(context: android.content.Context, bitmap: Bitmap) {
    val cachePath = File(context.cacheDir, "shared_collage.jpg")
    FileOutputStream(cachePath).use { bitmap.compress(Bitmap.CompressFormat.JPEG, 95, it) }
    val uri = androidx.core.content.FileProvider.getUriForFile(
        context, "${context.packageName}.fileprovider", cachePath
    )
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = "image/jpeg"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(intent, "Share collage"))
}
