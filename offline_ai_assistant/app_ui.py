"""
PySide6 desktop GUI for the Offline AI Assistant.

This module provides a cross-platform desktop interface with document upload,
search, and AI-powered response generation capabilities.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import traceback
import threading
import time
from datetime import datetime

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QLineEdit, QPushButton, QLabel, QFileDialog,
        QListWidget, QListWidgetItem, QSplitter, QTabWidget,
        QProgressBar, QProgressDialog, QStatusBar, QMenuBar, QMenu, QMessageBox,
        QDialog, QFormLayout, QSpinBox, QDoubleSpinBox, QComboBox,
        QCheckBox, QGroupBox, QScrollArea, QFrame, QTextBrowser
    )
    from PySide6.QtCore import (
        Qt, QThread, QObject, Signal, QTimer, QSettings, QSize
    )
    from PySide6.QtGui import QFont, QTextCursor, QIcon, QAction, QPixmap
except ImportError:
    print("PySide6 not installed. GUI will not work.")
    sys.exit(1)

from .config import Config, setup_logging
from .rag import RAGPipeline, create_rag_pipeline, ProcessingResult, RAGResult
from .llm import GenerationConfig
from .model_manager import ModelManager, ModelInfo

logger = logging.getLogger("OfflineAIAssistant.gui")


class WorkerSignals(QObject):
    """Signals for worker threads."""
    
    finished = Signal()
    error = Signal(str)
    progress = Signal(str)
    result = Signal(object)
    stream_token = Signal(str)
    stream_sources = Signal(list)
    stream_final = Signal(object)


class DocumentProcessor(QObject):
    """Worker for processing documents in background thread."""
    
    def __init__(self, rag_pipeline: RAGPipeline, file_paths: List[Path]):
        super().__init__()
        self.rag_pipeline = rag_pipeline
        self.file_paths = file_paths
        self.signals = WorkerSignals()
    
    def run(self):
        """Process documents."""
        try:
            results = []
            
            for i, file_path in enumerate(self.file_paths):
                self.signals.progress.emit(f"Processing {file_path.name} ({i+1}/{len(self.file_paths)})...")
                
                result = self.rag_pipeline.process_document(file_path)
                results.append(result)
                
                if not result.success:
                    self.signals.error.emit(f"Failed to process {file_path.name}: {result.error_message}")
            
            self.signals.result.emit(results)
            self.signals.finished.emit()
            
        except Exception as e:
            logger.error(f"Error in document processing: {e}")
            self.signals.error.emit(str(e))
            self.signals.finished.emit()


class QueryProcessor(QObject):
    """Worker for processing queries in background thread."""
    
    def __init__(self, rag_pipeline: RAGPipeline, query: str, template: str, streaming: bool = True):
        super().__init__()
        self.rag_pipeline = rag_pipeline
        self.query = query
        self.template = template
        self.streaming = streaming
        self.signals = WorkerSignals()
    
    def run(self):
        """Process query."""
        try:
            if self.streaming:
                for update in self.rag_pipeline.query_stream(self.query, self.template):
                    if update["type"] == "token":
                        self.signals.stream_token.emit(update["token"])
                    elif update["type"] == "sources":
                        self.signals.stream_sources.emit(update["sources"])
                    elif update["type"] == "final":
                        self.signals.stream_final.emit(update)
                    elif update["type"] == "error":
                        self.signals.error.emit(update["error"])
                        break
            else:
                result = self.rag_pipeline.query(self.query, self.template)
                self.signals.result.emit(result)
            
            self.signals.finished.emit()
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            self.signals.error.emit(str(e))
            self.signals.finished.emit()


class SettingsDialog(QDialog):
    """Settings configuration dialog."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.resize(500, 400)
        
        self.settings = {}
        self.setup_ui()
        self.load_settings()
    
    def setup_ui(self):
        """Setup the settings UI."""
        layout = QVBoxLayout(self)
        
        # Create form layout
        form_layout = QFormLayout()
        
        # Chunking settings
        chunk_group = QGroupBox("Text Chunking")
        chunk_layout = QFormLayout(chunk_group)
        
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(100, 2048)
        self.chunk_size_spin.setValue(Config.CHUNK_SIZE)
        chunk_layout.addRow("Chunk Size (tokens):", self.chunk_size_spin)
        
        self.chunk_overlap_spin = QSpinBox()
        self.chunk_overlap_spin.setRange(0, 200)
        self.chunk_overlap_spin.setValue(Config.CHUNK_OVERLAP)
        chunk_layout.addRow("Chunk Overlap (tokens):", self.chunk_overlap_spin)
        
        # Retrieval settings
        retrieval_group = QGroupBox("Retrieval")
        retrieval_layout = QFormLayout(retrieval_group)
        
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 20)
        self.top_k_spin.setValue(Config.TOP_K_RETRIEVAL)
        retrieval_layout.addRow("Top-K Results:", self.top_k_spin)
        
        # LLM settings
        llm_group = QGroupBox("Language Model")
        llm_layout = QFormLayout(llm_group)
        
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(100, 4096)
        self.max_tokens_spin.setValue(Config.LLM_MAX_TOKENS)
        llm_layout.addRow("Max Tokens:", self.max_tokens_spin)
        
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setValue(Config.LLM_TEMPERATURE)
        llm_layout.addRow("Temperature:", self.temperature_spin)
        
        self.top_p_spin = QDoubleSpinBox()
        self.top_p_spin.setRange(0.0, 1.0)
        self.top_p_spin.setSingleStep(0.1)
        self.top_p_spin.setValue(Config.LLM_TOP_P)
        llm_layout.addRow("Top-P:", self.top_p_spin)
        
        self.gpu_layers_spin = QSpinBox()
        self.gpu_layers_spin.setRange(0, 100)
        self.gpu_layers_spin.setValue(Config.LLM_N_GPU_LAYERS)
        llm_layout.addRow("GPU Layers:", self.gpu_layers_spin)
        
        # Model paths
        paths_group = QGroupBox("Model Paths")
        paths_layout = QFormLayout(paths_group)
        
        # Model manager button
        manage_models_button = QPushButton("Manage Models...")
        manage_models_button.clicked.connect(self.open_model_manager)
        paths_layout.addRow("", manage_models_button)
        
        self.llm_path_edit = QLineEdit()
        self.llm_path_edit.setText(str(Config.LLM_MODEL_PATH))
        llm_path_button = QPushButton("Browse...")
        llm_path_button.clicked.connect(self.browse_llm_model)
        llm_path_layout = QHBoxLayout()
        llm_path_layout.addWidget(self.llm_path_edit)
        llm_path_layout.addWidget(llm_path_button)
        paths_layout.addRow("LLM Model:", llm_path_layout)
        
        self.embedding_model_edit = QLineEdit()
        self.embedding_model_edit.setText(Config.EMBEDDING_MODEL_NAME)
        paths_layout.addRow("Embedding Model:", self.embedding_model_edit)
        
        # Add groups to layout
        layout.addWidget(chunk_group)
        layout.addWidget(retrieval_group)
        layout.addWidget(llm_group)
        layout.addWidget(paths_group)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_settings)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self.reset_settings)
        
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.save_button)
        
        layout.addLayout(button_layout)
    
    def browse_llm_model(self):
        """Browse for LLM model file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select LLM Model File",
            str(Config.MODELS_DIR),
            "GGUF Files (*.gguf);;All Files (*)"
        )
        
        if file_path:
            self.llm_path_edit.setText(file_path)
    
    def open_model_manager(self):
        """Open the model manager dialog."""
        dialog = ModelManagerDialog(self)
        if dialog.exec() == QDialog.Accepted:
            # Update the LLM path if a model was selected
            if hasattr(dialog, 'selected_model_path') and dialog.selected_model_path:
                self.llm_path_edit.setText(dialog.selected_model_path)
    
    def load_settings(self):
        """Load current settings."""
        current_settings = Config.get_settings_dict()
        
        self.chunk_size_spin.setValue(current_settings["chunk_size"])
        self.chunk_overlap_spin.setValue(current_settings["chunk_overlap"])
        self.top_k_spin.setValue(current_settings["top_k_retrieval"])
        self.max_tokens_spin.setValue(current_settings["llm_max_tokens"])
        self.temperature_spin.setValue(current_settings["llm_temperature"])
        self.top_p_spin.setValue(current_settings["llm_top_p"])
        self.gpu_layers_spin.setValue(current_settings["llm_n_gpu_layers"])
        self.llm_path_edit.setText(current_settings["llm_model_path"])
        self.embedding_model_edit.setText(current_settings["embedding_model"])
    
    def save_settings(self):
        """Save settings and close dialog."""
        self.settings = {
            "chunk_size": self.chunk_size_spin.value(),
            "chunk_overlap": self.chunk_overlap_spin.value(),
            "top_k_retrieval": self.top_k_spin.value(),
            "llm_max_tokens": self.max_tokens_spin.value(),
            "llm_temperature": self.temperature_spin.value(),
            "llm_top_p": self.top_p_spin.value(),
            "llm_n_gpu_layers": self.gpu_layers_spin.value(),
            "llm_model_path": self.llm_path_edit.text(),
            "embedding_model": self.embedding_model_edit.text()
        }
        
        Config.update_settings(self.settings)
        self.accept()
    
    def reset_settings(self):
        """Reset to default settings."""
        # Reset Config to defaults
        Config.reset_to_defaults()
        
        # Reload UI from Config
        self.chunk_size_spin.setValue(Config.CHUNK_SIZE)
        self.chunk_overlap_spin.setValue(Config.CHUNK_OVERLAP)
        self.top_k_spin.setValue(Config.TOP_K_RETRIEVAL)
        self.max_tokens_spin.setValue(Config.LLM_MAX_TOKENS)
        self.temperature_spin.setValue(Config.LLM_TEMPERATURE)
        self.top_p_spin.setValue(Config.LLM_TOP_P)
        self.gpu_layers_spin.setValue(Config.LLM_N_GPU_LAYERS)
        self.llm_path_edit.setText(str(Config.LLM_MODEL_PATH))
        self.embedding_model_edit.setText(Config.EMBEDDING_MODEL_NAME)


class ModelDownloadWorker(QObject):
    """Worker for downloading models in background thread."""
    
    def __init__(self, model_manager: ModelManager, model_id: str):
        super().__init__()
        self.model_manager = model_manager
        self.model_id = model_id
        self.signals = WorkerSignals()
    
    def run(self):
        """Download the model."""
        try:
            def progress_callback(downloaded, total):
                percentage = int((downloaded / total) * 100)
                self.signals.progress.emit(f"Downloading: {percentage}% ({downloaded / (1024*1024):.1f} / {total / (1024*1024):.1f} MB)")
            
            success = self.model_manager.download_model(self.model_id, progress_callback)
            
            if success:
                # Get the model path that was just downloaded
                model_path = self.model_manager.get_model_path(self.model_id)
                self.signals.result.emit({
                    "success": True, 
                    "model_id": self.model_id,
                    "model_path": str(model_path) if model_path else None
                })
            else:
                self.signals.error.emit("Failed to download model")
            
            self.signals.finished.emit()
            
        except Exception as e:
            logger.error(f"Error in model download: {e}")
            self.signals.error.emit(str(e))
            self.signals.finished.emit()


class ModelManagerDialog(QDialog):
    """Model management dialog for downloading and removing models."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Manager")
        self.setModal(True)
        self.resize(800, 600)
        
        self.model_manager = ModelManager()
        self.download_thread = None
        self.selected_model_path = None  # Track if a model was downloaded
        
        self.setup_ui()
        self.refresh_models()
    
    def setup_ui(self):
        """Setup the model manager UI."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("AI Model Manager")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Download and manage AI models for the assistant. Models are stored in ~/.config/ai-offline-assistant/models/")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Current model indicator
        current_model_group = QGroupBox("Current Model")
        current_model_layout = QVBoxLayout()
        
        current_model_path = Config.LLM_MODEL_PATH
        if current_model_path.exists():
            current_text = f"Currently using: <b>{current_model_path.name}</b>"
        else:
            current_text = "No model currently loaded"
        
        self.current_model_label = QLabel(current_text)
        self.current_model_label.setWordWrap(True)
        current_model_layout.addWidget(self.current_model_label)
        
        current_model_group.setLayout(current_model_layout)
        layout.addWidget(current_model_group)
        
        # Tabs for available and installed models
        tabs = QTabWidget()
        
        # Available models tab
        available_tab = QWidget()
        available_layout = QVBoxLayout(available_tab)
        
        available_label = QLabel("Available Models for Download:")
        available_label.setFont(QFont("Arial", 11, QFont.Bold))
        available_layout.addWidget(available_label)
        
        self.available_list = QListWidget()
        self.available_list.itemSelectionChanged.connect(self.on_available_selection_changed)
        available_layout.addWidget(self.available_list)
        
        # Model details
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(120)
        self.details_text.setReadOnly(True)
        available_layout.addWidget(self.details_text)
        
        # Download button
        download_layout = QHBoxLayout()
        self.download_button = QPushButton("Download Selected Model")
        self.download_button.clicked.connect(self.download_model)
        self.download_button.setEnabled(False)
        download_layout.addWidget(self.download_button)
        download_layout.addStretch()
        available_layout.addLayout(download_layout)
        
        tabs.addTab(available_tab, "Available Models")
        
        # Installed models tab
        installed_tab = QWidget()
        installed_layout = QVBoxLayout(installed_tab)
        
        installed_label = QLabel("Installed Models:")
        installed_label.setFont(QFont("Arial", 11, QFont.Bold))
        installed_layout.addWidget(installed_label)
        
        self.installed_list = QListWidget()
        self.installed_list.itemSelectionChanged.connect(self.on_installed_selection_changed)
        installed_layout.addWidget(self.installed_list)
        
        # Installed model info
        self.installed_info_text = QTextEdit()
        self.installed_info_text.setMaximumHeight(100)
        self.installed_info_text.setReadOnly(True)
        installed_layout.addWidget(self.installed_info_text)
        
        # Action buttons
        installed_actions_layout = QHBoxLayout()
        
        self.use_model_button = QPushButton("Use This Model")
        self.use_model_button.clicked.connect(self.use_selected_model)
        self.use_model_button.setEnabled(False)
        installed_actions_layout.addWidget(self.use_model_button)
        
        self.delete_button = QPushButton("Delete Model")
        self.delete_button.clicked.connect(self.delete_model)
        self.delete_button.setEnabled(False)
        installed_actions_layout.addWidget(self.delete_button)
        
        installed_actions_layout.addStretch()
        installed_layout.addLayout(installed_actions_layout)
        
        tabs.addTab(installed_tab, "Installed Models")
        
        layout.addWidget(tabs)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        # Close button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
    
    def refresh_models(self):
        """Refresh the model lists."""
        # Refresh available models
        self.available_list.clear()
        available_models = self.model_manager.get_available_models()
        
        for model_info in available_models:
            item = QListWidgetItem()
            is_installed = self.model_manager.is_model_installed(model_info.id)
            status = " [INSTALLED]" if is_installed else ""
            item.setText(f"{model_info.name} ({model_info.size_gb:.1f} GB){status}")
            item.setData(Qt.UserRole, model_info)
            self.available_list.addItem(item)
        
        # Refresh installed models
        self.installed_list.clear()
        installed_models = self.model_manager.get_installed_models()
        
        current_model_name = Config.LLM_MODEL_PATH.name if Config.LLM_MODEL_PATH.exists() else None
        
        for model in installed_models:
            item = QListWidgetItem()
            is_current = model['filename'] == current_model_name
            status = " [ACTIVE]" if is_current else ""
            item.setText(f"{model['name']} ({model['size_mb']:.1f} MB){status}")
            item.setData(Qt.UserRole, model)
            self.installed_list.addItem(item)
        
        if not installed_models:
            no_models_item = QListWidgetItem("No models installed yet")
            no_models_item.setFlags(Qt.NoItemFlags)
            self.installed_list.addItem(no_models_item)
    
    def on_available_selection_changed(self):
        """Handle available model selection change."""
        current_item = self.available_list.currentItem()
        if current_item and current_item.data(Qt.UserRole):
            model_info = current_item.data(Qt.UserRole)
            is_installed = self.model_manager.is_model_installed(model_info.id)
            
            details = f"<b>{model_info.name}</b><br>"
            details += f"{model_info.description}<br><br>"
            details += f"<b>Size:</b> {model_info.size_gb:.1f} GB<br>"
            details += f"<b>Quantization:</b> {model_info.quantization}<br>"
            details += f"<b>Context Length:</b> {model_info.context_length:,} tokens<br>"
            details += f"<b>Status:</b> {'Installed' if is_installed else 'Not installed'}"
            
            self.details_text.setHtml(details)
            self.download_button.setEnabled(not is_installed)
        else:
            self.details_text.clear()
            self.download_button.setEnabled(False)
    
    def on_installed_selection_changed(self):
        """Handle installed model selection change."""
        current_item = self.installed_list.currentItem()
        if current_item and current_item.data(Qt.UserRole):
            model = current_item.data(Qt.UserRole)
            
            # Check if this is the currently active model
            current_model_name = Config.LLM_MODEL_PATH.name if Config.LLM_MODEL_PATH.exists() else None
            is_current = model['filename'] == current_model_name
            
            info = f"<b>{model['name']}</b><br>"
            info += f"{model['description']}<br><br>"
            info += f"<b>File:</b> {model['filename']}<br>"
            info += f"<b>Size:</b> {model['size_mb']:.1f} MB<br>"
            info += f"<b>Downloaded:</b> {model['downloaded_at']}<br>"
            
            if is_current:
                info += "<br><b style='color: green;'>[ACTIVE] This is the currently active model</b>"
            
            self.installed_info_text.setHtml(info)
            self.delete_button.setEnabled(True)
            
            # Update button text based on whether it's current
            if is_current:
                self.use_model_button.setText("Already Active")
                self.use_model_button.setEnabled(False)
            else:
                self.use_model_button.setText("Switch to This Model")
                self.use_model_button.setEnabled(True)
        else:
            self.installed_info_text.clear()
            self.delete_button.setEnabled(False)
            self.use_model_button.setEnabled(False)
            self.use_model_button.setText("Use This Model")
    
    def download_model(self):
        """Download the selected model."""
        current_item = self.available_list.currentItem()
        if not current_item or not current_item.data(Qt.UserRole):
            return
        
        model_info = current_item.data(Qt.UserRole)
        
        # Confirm download
        reply = QMessageBox.question(
            self,
            "Confirm Download",
            f"Download {model_info.name}?\n\nSize: {model_info.size_gb:.1f} GB\n\nThis may take a while depending on your internet connection.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply != QMessageBox.Yes:
            return
        
        # Start download
        self.download_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText(f"Downloading {model_info.name}...")
        
        # Create worker and thread
        self.download_worker = ModelDownloadWorker(self.model_manager, model_info.id)
        self.download_thread = QThread()
        
        # Move worker to thread
        self.download_worker.moveToThread(self.download_thread)
        
        # Connect signals
        self.download_thread.started.connect(self.download_worker.run)
        self.download_worker.signals.progress.connect(self.update_download_progress)
        self.download_worker.signals.error.connect(self.on_download_error)
        self.download_worker.signals.result.connect(self.on_download_complete)
        self.download_worker.signals.finished.connect(self.download_thread.quit)
        self.download_worker.signals.finished.connect(self.download_worker.deleteLater)
        self.download_thread.finished.connect(self.download_thread.deleteLater)
        
        # Start download
        self.download_thread.start()
    
    def update_download_progress(self, message: str):
        """Update download progress message."""
        self.status_label.setText(message)
    
    def on_download_complete(self, result: Dict):
        """Handle download completion."""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Model downloaded successfully!")
        self.download_button.setEnabled(True)
        
        # Set the selected model path so the main window can use it
        if result and 'model_path' in result:
            self.selected_model_path = result['model_path']
            logger.info(f"Model downloaded to: {self.selected_model_path}")
        
        # Refresh lists
        self.refresh_models()
        
        QMessageBox.information(
            self,
            "Download Complete",
            f"Model downloaded successfully!\n\n"
            f"Please close the Model Manager window to load the model.\n"
            f"This may take 10-30 seconds depending on your system."
        )
    
    def on_download_error(self, error: str):
        """Handle download error."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Download failed")
        self.download_button.setEnabled(True)
        
        QMessageBox.critical(
            self,
            "Download Error",
            f"Failed to download model:\n\n{error}"
        )
    
    def delete_model(self):
        """Delete the selected model."""
        current_item = self.installed_list.currentItem()
        if not current_item or not current_item.data(Qt.UserRole):
            return
        
        model = current_item.data(Qt.UserRole)
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Delete {model['name']}?\n\nThis will permanently remove the model file ({model['size_mb']:.1f} MB).",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            success = self.model_manager.delete_model(model['filename'])
            
            if success:
                self.status_label.setText(f"Model deleted: {model['name']}")
                self.refresh_models()
            else:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to delete model: {model['name']}"
                )
    
    def use_selected_model(self):
        """Set the selected model as the active model."""
        current_item = self.installed_list.currentItem()
        if not current_item or not current_item.data(Qt.UserRole):
            return
        
        model = current_item.data(Qt.UserRole)
        
        # Check if it's already the current model
        current_model_name = Config.LLM_MODEL_PATH.name if Config.LLM_MODEL_PATH.exists() else None
        if model['filename'] == current_model_name:
            QMessageBox.information(
                self,
                "Already Active",
                f"{model['name']} is already the active model."
            )
            return
        
        self.selected_model_path = model['path']
        
        # Update the current model label
        self.current_model_label.setText(f"Currently using: <b>{model['filename']}</b> (will load on close)")
        
        QMessageBox.information(
            self,
            "Model Selected",
            f"Model selected: {model['name']}\n\n"
            f"Close this window to load the model.\n"
            f"This may take 10-30 seconds depending on your system."
        )
        
        # Don't auto-close, let user close when ready
        # self.accept()


class DocumentListWidget(QListWidget):
    """Custom list widget for documents with enhanced display."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.documents = []
    
    def update_documents(self, documents: List[Dict[str, Any]]):
        """Update the document list."""
        self.clear()
        self.documents = documents
        
        for doc in documents:
            item = QListWidgetItem()
            
            # Format document info
            name = doc["file_name"]
            size_mb = doc["file_size"] / (1024 * 1024)
            chunks = doc["chunk_count"]
            created = datetime.fromisoformat(doc["created_at"]).strftime("%Y-%m-%d %H:%M")
            
            item.setText(f"{name}")
            item.setToolTip(f"Size: {size_mb:.1f} MB\nChunks: {chunks}\nCreated: {created}")
            item.setData(Qt.UserRole, doc["document_id"])
            
            self.addItem(item)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.rag_pipeline = None
        self.current_query_thread = None
        self.current_processing_thread = None
        
        # Settings
        self.settings = QSettings("OfflineAIAssistant", "MainApp")
        
        self.setup_ui()
        self.setup_menu()
        self.setup_status_bar()
        self.load_window_settings()
        
        # Initialize RAG pipeline
        self.initialize_rag_pipeline()
        
        # Refresh document list
        self.refresh_documents()
    
    def setup_ui(self):
        """Setup the main UI."""
        self.setWindowTitle("Offline AI Assistant")
        self.setMinimumSize(Config.WINDOW_MIN_WIDTH, Config.WINDOW_MIN_HEIGHT)
        self.resize(Config.WINDOW_WIDTH, Config.WINDOW_HEIGHT)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Create splitter
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel - Documents
        left_panel = self.create_documents_panel()
        splitter.addWidget(left_panel)
        
        # Right panel - Chat
        right_panel = self.create_chat_panel()
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([300, 900])
    
    def create_documents_panel(self) -> QWidget:
        """Create the documents panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("Documents")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(title)
        
        # Upload button
        self.upload_button = QPushButton("Upload Documents")
        self.upload_button.clicked.connect(self.upload_documents)
        layout.addWidget(self.upload_button)
        
        # Document list
        self.document_list = DocumentListWidget()
        self.document_list.itemSelectionChanged.connect(self.on_document_selected)
        layout.addWidget(self.document_list)
        
        # Document actions
        doc_actions_layout = QHBoxLayout()
        
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_documents)
        doc_actions_layout.addWidget(self.refresh_button)
        
        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self.delete_selected_document)
        self.delete_button.setEnabled(False)
        doc_actions_layout.addWidget(self.delete_button)
        
        layout.addLayout(doc_actions_layout)
        
        # Statistics
        self.stats_label = QLabel("No documents loaded")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)
        
        return panel
    
    def create_chat_panel(self) -> QWidget:
        """Create the chat panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title and template selection
        header_layout = QHBoxLayout()
        
        title = QLabel("AI Assistant")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # Template selection
        template_label = QLabel("Template:")
        header_layout.addWidget(template_label)
        
        self.template_combo = QComboBox()
        self.template_combo.addItems(list(Config.PROMPT_TEMPLATES.keys()))
        header_layout.addWidget(self.template_combo)
        
        layout.addLayout(header_layout)
        
        # Chat display
        self.chat_display = QTextBrowser()
        self.chat_display.setFont(QFont("Consolas", 10))
        layout.addWidget(self.chat_display)
        
        # Sources panel
        sources_label = QLabel("Sources:")
        sources_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(sources_label)
        
        self.sources_display = QTextEdit()
        self.sources_display.setMaximumHeight(150)
        self.sources_display.setReadOnly(True)
        layout.addWidget(self.sources_display)
        
        # Query input
        query_layout = QHBoxLayout()
        
        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Ask a question about your documents...")
        self.query_input.returnPressed.connect(self.process_query)
        query_layout.addWidget(self.query_input)
        
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.process_query)
        query_layout.addWidget(self.send_button)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_chat)
        query_layout.addWidget(self.clear_button)
        
        layout.addLayout(query_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        return panel
    
    def setup_menu(self):
        """Setup the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        upload_action = QAction("Upload Documents", self)
        upload_action.triggered.connect(self.upload_documents)
        file_menu.addAction(upload_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        model_manager_action = QAction("Manage Models...", self)
        model_manager_action.triggered.connect(self.show_model_manager)
        edit_menu.addAction(model_manager_action)
        
        edit_menu.addSeparator()
        
        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(self.show_settings)
        edit_menu.addAction(settings_action)
        
        clear_action = QAction("Clear Chat", self)
        clear_action.triggered.connect(self.clear_chat)
        edit_menu.addAction(clear_action)
        
        # View menu
        view_menu = menubar.addMenu("View")
        
        refresh_action = QAction("Refresh Documents", self)
        refresh_action.triggered.connect(self.refresh_documents)
        view_menu.addAction(refresh_action)
        
        stats_action = QAction("Show Statistics", self)
        stats_action.triggered.connect(self.show_statistics)
        view_menu.addAction(stats_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Setup the status bar."""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
        
        # Add permanent widgets
        self.model_status_label = QLabel("Model: Not loaded")
        self.status_bar.addPermanentWidget(self.model_status_label)
    
    def initialize_rag_pipeline(self):
        """Initialize the RAG pipeline."""
        try:
            self.status_bar.showMessage("Initializing AI models...")
            
            # Check if model file exists
            model_path = Config.LLM_MODEL_PATH
            logger.info(f"Looking for LLM model at: {model_path}")
            
            if not model_path.exists():
                error_msg = f"LLM model file not found at: {model_path}\n\nPlease use Edit → Manage Models to download a model, or update the path in Settings."
                self.show_error("Model Not Found", error_msg)
                self.model_status_label.setText("Model: Not found")
                self.status_bar.showMessage("Model not found - Use Edit → Manage Models to download")
                return
            
            logger.info(f"Model file found: {model_path} ({model_path.stat().st_size / (1024**3):.2f} GB)")
            
            # Create RAG pipeline
            self.rag_pipeline = create_rag_pipeline(
                model_path=model_path,
                embedding_model=Config.EMBEDDING_MODEL_NAME
            )
            
            # Update model status
            if self.rag_pipeline.llm and self.rag_pipeline.llm.is_loaded():
                model_name = self.rag_pipeline.llm.model_path.name
                self.model_status_label.setText(f"Model: {model_name}")
                logger.info(f"LLM model loaded successfully: {model_name}")
            else:
                error_msg = "LLM model failed to load. This could be due to:\n\n• Insufficient RAM (need 8GB+ free)\n• Corrupted model file\n• Incompatible model format\n\nTry a smaller model or check the logs for details."
                self.show_error("Model Load Failed", error_msg)
                self.model_status_label.setText("Model: Load failed")
                self.status_bar.showMessage("Model load failed")
                return
            
            self.status_bar.showMessage("Ready")
            
        except Exception as e:
            logger.error(f"Error initializing RAG pipeline: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.show_error("Initialization Error", f"Failed to initialize AI models:\n\n{str(e)}\n\nCheck the logs for more details.")
            self.model_status_label.setText("Model: Error")
    
    def upload_documents(self):
        """Upload documents for processing."""
        # Open file dialog in user's home directory
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Documents to Upload",
            str(Path.home()),
            "Documents (*.pdf *.docx);;PDF Files (*.pdf);;Word Documents (*.docx);;All Files (*)"
        )
        
        if not file_paths:
            return
        
        if not self.rag_pipeline:
            self.show_error("Error", "RAG pipeline not initialized")
            return
        
        # Convert to Path objects
        paths = [Path(p) for p in file_paths]
        
        # Start processing in background thread
        self.start_document_processing(paths)
    
    def start_document_processing(self, file_paths: List[Path]):
        """Start document processing in background thread."""
        # Check if thread exists and is running
        try:
            if (self.current_processing_thread and 
                hasattr(self.current_processing_thread, 'isRunning') and 
                self.current_processing_thread.isRunning()):
                self.show_error("Error", "Document processing already in progress")
                return
        except RuntimeError:
            # Thread object was deleted, reset it
            self.current_processing_thread = None
        
        # Setup progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.upload_button.setEnabled(False)
        
        # Create worker and thread
        self.processing_worker = DocumentProcessor(self.rag_pipeline, file_paths)
        self.current_processing_thread = QThread()
        
        # Move worker to thread
        self.processing_worker.moveToThread(self.current_processing_thread)
        
        # Connect signals
        self.current_processing_thread.started.connect(self.processing_worker.run)
        self.processing_worker.signals.progress.connect(self.update_progress)
        self.processing_worker.signals.error.connect(self.show_processing_error)
        self.processing_worker.signals.result.connect(self.on_processing_complete)
        self.processing_worker.signals.finished.connect(self.current_processing_thread.quit)
        self.processing_worker.signals.finished.connect(self.processing_worker.deleteLater)
        self.current_processing_thread.finished.connect(self._on_processing_thread_finished)
        self.current_processing_thread.finished.connect(self.current_processing_thread.deleteLater)
        
        # Start thread
        self.current_processing_thread.start()
    
    def update_progress(self, message: str):
        """Update progress message."""
        self.status_bar.showMessage(message)
    
    def show_processing_error(self, error: str):
        """Show processing error."""
        self.show_error("Processing Error", error)
    
    def on_processing_complete(self, results: List[ProcessingResult]):
        """Handle processing completion."""
        self.progress_bar.setVisible(False)
        self.upload_button.setEnabled(True)
        
        # Show results
        success_count = sum(1 for r in results if r.success)
        total_count = len(results)
        
        if success_count == total_count:
            self.status_bar.showMessage(f"Successfully processed {success_count} documents")
        else:
            failed_count = total_count - success_count
            self.status_bar.showMessage(f"Processed {success_count}/{total_count} documents ({failed_count} failed)")
            
            # Show detailed error information
            failed_files = [r.file_path for r in results if not r.success]
            if failed_files:
                error_msg = f"Failed to process:\n" + "\n".join([Path(f).name for f in failed_files[:5]])
                if len(failed_files) > 5:
                    error_msg += f"\n... and {len(failed_files) - 5} more files"
                self.show_error("Processing Errors", error_msg)
        
        # Refresh document list
        self.refresh_documents()
    
    def _on_processing_thread_finished(self):
        """Handle processing thread finished signal."""
        # Clean up thread reference with a small delay to ensure proper cleanup
        QTimer.singleShot(100, self._clear_processing_thread_reference)
    
    def _clear_processing_thread_reference(self):
        """Clear the processing thread reference."""
        self.current_processing_thread = None
    
    def process_query(self):
        """Process user query."""
        query = self.query_input.text().strip()
        if not query:
            return
        
        if not self.rag_pipeline:
            self.show_error("Error", "RAG pipeline not initialized")
            return
        
        if not self.rag_pipeline.llm or not self.rag_pipeline.llm.is_loaded():
            self.show_error("Error", "Language model not loaded")
            return
        
        # Add query to chat
        self.add_to_chat(f"**You:** {query}", is_user=True)
        
        # Clear input
        self.query_input.clear()
        
        # Start query processing
        self.start_query_processing(query)
    
    def start_query_processing(self, query: str):
        """Start query processing in background thread."""
        # Check if thread exists and is running
        try:
            if (self.current_query_thread and 
                hasattr(self.current_query_thread, 'isRunning') and 
                self.current_query_thread.isRunning()):
                self.show_error("Error", "Query processing already in progress")
                return
        except RuntimeError:
            # Thread object was deleted, reset it
            self.current_query_thread = None
        
        # Disable input
        self.send_button.setEnabled(False)
        self.query_input.setEnabled(False)
        
        # Get selected template
        template = self.template_combo.currentText()
        
        # Create worker and thread
        self.query_worker = QueryProcessor(self.rag_pipeline, query, template, streaming=True)
        self.current_query_thread = QThread()
        
        # Move worker to thread
        self.query_worker.moveToThread(self.current_query_thread)
        
        # Connect signals
        self.current_query_thread.started.connect(self.query_worker.run)
        self.query_worker.signals.stream_token.connect(self.on_stream_token)
        self.query_worker.signals.stream_sources.connect(self.on_stream_sources)
        self.query_worker.signals.stream_final.connect(self.on_stream_final)
        self.query_worker.signals.error.connect(self.show_query_error)
        self.query_worker.signals.finished.connect(self.on_query_finished)
        self.query_worker.signals.finished.connect(self.current_query_thread.quit)
        self.query_worker.signals.finished.connect(self.query_worker.deleteLater)
        self.current_query_thread.finished.connect(self._on_query_thread_finished)
        self.current_query_thread.finished.connect(self.current_query_thread.deleteLater)
        
        # Start thread
        self.current_query_thread.start()
        
        # Add assistant response placeholder
        self.add_to_chat("**Assistant:** ", is_user=False)
        self.current_response_start = self.chat_display.textCursor().position()
    
    def on_stream_token(self, token: str):
        """Handle streaming token."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(token)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()
    
    def on_stream_sources(self, sources: List[Dict[str, Any]]):
        """Handle streaming sources."""
        sources_text = ""
        for source in sources:
            sources_text += f"[{source['rank']}] {source['file_name']} (Score: {source['score']:.3f})\n"
            sources_text += f"    {source['text_preview']}\n\n"
        
        self.sources_display.setText(sources_text)
    
    def on_stream_final(self, result: Dict[str, Any]):
        """Handle final streaming result."""
        # Add timing information
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        # Build timing info string with safe key access
        total_time = result.get('total_time', 0.0)
        tokens_generated = result.get('tokens_generated', 0)
        chunks_retrieved = result.get('chunks_retrieved', 0)
        
        timing_info = f"\n\n*Generated in {total_time:.2f}s"
        if tokens_generated > 0:
            timing_info += f" ({tokens_generated} tokens"
            if chunks_retrieved > 0:
                timing_info += f", {chunks_retrieved} sources"
            timing_info += ")"
        elif chunks_retrieved > 0:
            timing_info += f" ({chunks_retrieved} sources)"
        timing_info += "*"
        
        cursor.insertText(timing_info)
        self.chat_display.setTextCursor(cursor)
    
    def show_query_error(self, error: str):
        """Show query error."""
        self.add_to_chat(f"**Error:** {error}", is_user=False)
        self.show_error("Query Error", error)
    
    def on_query_finished(self):
        """Handle query completion."""
        self.send_button.setEnabled(True)
        self.query_input.setEnabled(True)
        self.query_input.setFocus()
    
    def _on_query_thread_finished(self):
        """Handle query thread finished signal."""
        # Clean up thread reference with a small delay to ensure proper cleanup
        QTimer.singleShot(100, self._clear_query_thread_reference)
    
    def _clear_query_thread_reference(self):
        """Clear the query thread reference."""
        self.current_query_thread = None
    
    def add_to_chat(self, message: str, is_user: bool = False):
        """Add message to chat display."""
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        if cursor.position() > 0:
            cursor.insertText("\n\n")
        
        cursor.insertText(message)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()
    
    def clear_chat(self):
        """Clear the chat display."""
        self.chat_display.clear()
        self.sources_display.clear()
    
    def refresh_documents(self):
        """Refresh the document list."""
        if not self.rag_pipeline:
            return
        
        try:
            documents = self.rag_pipeline.list_documents()
            self.document_list.update_documents(documents)
            
            # Update statistics
            total_docs = len(documents)
            total_chunks = sum(doc["chunk_count"] for doc in documents)
            
            if total_docs > 0:
                self.stats_label.setText(f"Documents: {total_docs}\nTotal chunks: {total_chunks}")
            else:
                self.stats_label.setText("No documents loaded")
                
        except Exception as e:
            logger.error(f"Error refreshing documents: {e}")
            self.show_error("Error", f"Failed to refresh documents: {str(e)}")
    
    def on_document_selected(self):
        """Handle document selection."""
        current_item = self.document_list.currentItem()
        self.delete_button.setEnabled(current_item is not None)
    
    def delete_selected_document(self):
        """Delete the selected document."""
        current_item = self.document_list.currentItem()
        if not current_item:
            return
        
        document_id = current_item.data(Qt.UserRole)
        document_name = current_item.text()
        
        # Confirm deletion
        reply = QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete '{document_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            try:
                success = self.rag_pipeline.delete_document(document_id)
                if success:
                    self.refresh_documents()
                    self.status_bar.showMessage(f"Deleted document: {document_name}")
                else:
                    self.show_error("Error", "Failed to delete document")
            except Exception as e:
                logger.error(f"Error deleting document: {e}")
                self.show_error("Error", f"Failed to delete document: {str(e)}")
    
    def show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.Accepted:
            # Reinitialize pipeline with new settings
            self.initialize_rag_pipeline()
            self.status_bar.showMessage("Settings updated")
    
    def show_model_manager(self):
        """Show model manager dialog."""
        dialog = ModelManagerDialog(self)
        dialog.exec()
        
        # After dialog closes, check if a model was downloaded or selected
        if hasattr(dialog, 'selected_model_path') and dialog.selected_model_path:
            logger.info(f"Model downloaded/selected: {dialog.selected_model_path}")
            # Update the config with the new model path
            Config.LLM_MODEL_PATH = Path(dialog.selected_model_path)
            
            # Show progress dialog while loading model
            progress = QProgressDialog(self)
            progress.setWindowTitle("Loading Model")
            progress.setWindowModality(Qt.WindowModal)
            progress.setCancelButton(None)  # No cancel button
            progress.setMinimumDuration(0)  # Show immediately
            progress.setRange(0, 4)  # 4 steps
            progress.setValue(0)
            progress.setLabelText("Step 1/4: Verifying model file...")
            progress.show()
            QApplication.processEvents()  # Force UI update
            
            # Reinitialize the RAG pipeline with the new model
            try:
                # Step 1: Verify model exists
                progress.setLabelText("Step 1/4: Verifying model file...")
                QApplication.processEvents()
                
                # Step 2: Load embedding model
                progress.setValue(1)
                progress.setLabelText("Step 2/4: Loading embedding model...")
                QApplication.processEvents()
                
                # Step 3: Load LLM model (this is the slow part)
                progress.setValue(2)
                progress.setLabelText("Step 3/4: Loading LLM model (this may take 10-30s)...")
                QApplication.processEvents()
                
                # Actually initialize
                self.initialize_rag_pipeline()
                
                # Step 4: Complete
                progress.setValue(3)
                progress.setLabelText("Step 4/4: Finalizing...")
                QApplication.processEvents()
                
                progress.setValue(4)
                progress.close()
                
                self.status_bar.showMessage("Model loaded successfully!")
                QMessageBox.information(
                    self,
                    "Model Ready",
                    f"Model loaded successfully!\n\nYou can now upload documents and start asking questions."
                )
            except Exception as e:
                progress.close()
                logger.error(f"Failed to initialize RAG pipeline: {e}")
                QMessageBox.critical(
                    self,
                    "Model Load Failed",
                    f"Failed to load model:\n{str(e)}"
                )
    
    def show_statistics(self):
        """Show application statistics."""
        if not self.rag_pipeline:
            return
        
        try:
            stats = self.rag_pipeline.get_statistics()
            
            stats_text = f"""
Pipeline Statistics:
- Documents processed: {stats['pipeline_stats']['documents_processed']}
- Chunks created: {stats['pipeline_stats']['chunks_created']}
- Queries answered: {stats['pipeline_stats']['queries_answered']}
- Average processing time: {stats['avg_processing_time']:.2f}s
- Average query time: {stats['avg_query_time']:.2f}s

Vector Store:
- Documents: {stats['vector_store']['documents']}
- Chunks: {stats['vector_store']['chunks']}
- Vectors in index: {stats['vector_store']['vectors_in_index']}
- Database size: {stats['vector_store']['database_size_bytes'] / (1024*1024):.1f} MB
- Index size: {stats['vector_store']['index_size_bytes'] / (1024*1024):.1f} MB

Language Model:
- Status: {stats['llm']['status']}
- Model: {stats['llm'].get('model_path', 'N/A')}
- Context length: {stats['llm'].get('context_length', 'N/A')}
- GPU layers: {stats['llm'].get('gpu_layers', 'N/A')}

Embedder:
- Status: {stats['embedder']['status']}
- Model: {stats['embedder'].get('model_name', 'N/A')}
- Dimension: {stats['embedder'].get('embedding_dim', 'N/A')}
"""
            
            QMessageBox.information(self, "Statistics", stats_text)
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            self.show_error("Error", f"Failed to get statistics: {str(e)}")
    
    def show_about(self):
        """Show about dialog."""
        about_text = """
<h2>Offline AI Assistant</h2>
<p>A fully offline desktop AI assistant for document analysis and question answering.</p>

<h3>Features:</h3>
<ul>
<li>Upload and parse PDF and DOCX files</li>
<li>Semantic search across documents</li>
<li>AI-powered responses with citations</li>
<li>Fully offline operation</li>
<li>Cross-platform compatibility</li>
</ul>

<h3>Technology Stack:</h3>
<ul>
<li>Python 3.10+</li>
<li>PySide6 (Qt for Python)</li>
<li>FAISS for vector search</li>
<li>sentence-transformers for embeddings</li>
<li>llama-cpp-python for LLM inference</li>
<li>SQLite for metadata storage</li>
</ul>

<p><b>Privacy:</b> All data stays on your local machine. No external API calls are made.</p>
"""
        
        QMessageBox.about(self, "About Offline AI Assistant", about_text)
    
    def show_error(self, title: str, message: str):
        """Show error message."""
        QMessageBox.critical(self, title, message)
    
    def load_window_settings(self):
        """Load window settings."""
        self.restoreGeometry(self.settings.value("geometry", self.saveGeometry()))
        self.restoreState(self.settings.value("windowState", self.saveState()))
    
    def save_window_settings(self):
        """Save window settings."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
    
    def closeEvent(self, event):
        """Handle close event."""
        # Save settings
        self.save_window_settings()
        
        # Stop any running threads properly
        self._cleanup_threads()
        
        # Close RAG pipeline and cleanup resources
        if self.rag_pipeline:
            try:
                # Force cleanup of sentence-transformers resources
                if hasattr(self.rag_pipeline.embedder, 'model') and self.rag_pipeline.embedder.model:
                    # Clear the model to free up resources
                    del self.rag_pipeline.embedder.model
                    self.rag_pipeline.embedder.model = None
                
                self.rag_pipeline.close()
            except Exception as e:
                logger.error(f"Error during RAG pipeline cleanup: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        event.accept()
    
    def _cleanup_threads(self):
        """Properly cleanup all running threads."""
        threads_to_cleanup = []
        
        # Collect threads that need cleanup
        if self.current_query_thread:
            try:
                if hasattr(self.current_query_thread, 'isRunning') and self.current_query_thread.isRunning():
                    threads_to_cleanup.append(('query', self.current_query_thread))
            except RuntimeError:
                pass
        
        if self.current_processing_thread:
            try:
                if hasattr(self.current_processing_thread, 'isRunning') and self.current_processing_thread.isRunning():
                    threads_to_cleanup.append(('processing', self.current_processing_thread))
            except RuntimeError:
                pass
        
        # Stop threads gracefully
        for thread_name, thread in threads_to_cleanup:
            try:
                logger.info(f"Stopping {thread_name} thread...")
                
                # First, try to quit gracefully
                thread.quit()
                
                # Wait for thread to finish with longer timeout
                if not thread.wait(5000):  # Wait up to 5 seconds
                    logger.warning(f"{thread_name} thread did not stop gracefully, terminating...")
                    thread.terminate()
                    if not thread.wait(2000):  # Wait up to 2 seconds for termination
                        logger.error(f"{thread_name} thread could not be terminated")
                    else:
                        logger.info(f"{thread_name} thread terminated")
                else:
                    logger.info(f"{thread_name} thread stopped gracefully")
                
            except Exception as e:
                logger.error(f"Error stopping {thread_name} thread: {e}")
        
        # Process any pending events to ensure cleanup signals are handled
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app:
            app.processEvents()
        
        # Clear thread references
        self.current_query_thread = None
        self.current_processing_thread = None


def main():
    """Main application entry point."""
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Offline AI Assistant")
    
    # Ensure directories exist
    Config.ensure_directories()
    
    # Load saved configuration (if exists)
    Config.load_config()
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Offline AI Assistant")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("OfflineAIAssistant")
    
    # Set application properties for better resource management
    app.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings, True)
    
    window = None
    try:
        # Create main window
        window = MainWindow()
        window.show()
        
        # Run application
        exit_code = app.exec()
        
        # Ensure proper cleanup
        if window:
            window._cleanup_threads()
        
        logger.info("Application shutting down")
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        
        # Cleanup on error
        if window:
            try:
                window._cleanup_threads()
            except:
                pass
        
        # Show error dialog
        error_msg = f"A fatal error occurred:\n\n{str(e)}\n\nCheck the logs for more details."
        QMessageBox.critical(None, "Fatal Error", error_msg)
        sys.exit(1)
    finally:
        # Final cleanup
        try:
            app.quit()
        except:
            pass


if __name__ == "__main__":
    main()
