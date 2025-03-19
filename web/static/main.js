// Panel Collapsible Functionality
document.addEventListener('DOMContentLoaded', function() {
  // Initialize save note button
  const saveNoteBtn = document.getElementById('save-note-btn');
  if (saveNoteBtn) {
    saveNoteBtn.onclick = saveManualNote;
  }
  
  // Get all collapsible panels
  const panels = document.querySelectorAll('.panel-toggle');
  
  // Add click event listeners to each panel toggle button
  panels.forEach(panel => {
    panel.addEventListener('click', function(e) {
      e.preventDefault();
      
      // Get the target content element
      const targetId = this.getAttribute('data-target').substring(1);
      const contentElement = document.getElementById(targetId);
      
      // Toggle the content visibility
      contentElement.classList.toggle('hidden');
      
      // Update the expanded state
      const isExpanded = !contentElement.classList.contains('hidden');
      this.setAttribute('aria-expanded', isExpanded);
      
      // Update the chevron icon
      const icon = this.querySelector('.fa-chevron-down');
      if (isExpanded) {
        icon.style.transform = 'rotate(0deg)';
      } else {
        icon.style.transform = 'rotate(-90deg)';
      }
    });
  });
  
  // Initialize all panels as expanded by default
  panels.forEach(panel => {
    // Set aria-expanded to true
    panel.setAttribute('aria-expanded', 'true');
    
    // Make sure the chevron is pointing down
    const icon = panel.querySelector('.fa-chevron-down');
    icon.style.transform = 'rotate(0deg)';
    
    // Make sure the content is visible
    const targetId = panel.getAttribute('data-target').substring(1);
    const contentElement = document.getElementById(targetId);
    if (contentElement) {
      contentElement.classList.remove('hidden');
    }
  });
  
  // Tab functionality
  const tabButtons = document.querySelectorAll('[role="tab"]');
  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const tabId = button.getAttribute('data-tab-target');
      
      // Hide all tab panes
      document.querySelectorAll('.tab-pane').forEach(pane => {
        pane.classList.remove('active');
        pane.style.display = 'none';
      });
      
      // Deactivate all tab buttons
      document.querySelectorAll('[role="tab"]').forEach(btn => {
        btn.classList.remove('active');
        btn.classList.remove('border-b-2');
        btn.classList.remove('border-primary-500');
        btn.classList.remove('bg-dark-800');
      });
      
      // Show the selected tab pane
      const selectedPane = document.getElementById(tabId);
      selectedPane.classList.add('active');
      selectedPane.style.display = 'block';
      
      // Activate the clicked tab button
      button.classList.add('active');
      button.classList.add('border-b-2');
      button.classList.add('border-primary-500');
      button.classList.add('bg-dark-800');
    });
  });
  
  // Make sure the first tab is active on page load
  const firstTabPane = document.querySelector('.tab-pane');
  if (firstTabPane) {
    firstTabPane.style.display = 'block';
  }
  
  // Setup video browser modal
  const showVideosBtn = document.getElementById('show-videos-btn');
  if (showVideosBtn) {
    showVideosBtn.addEventListener('click', () => {
      // Open modal
      const modal = document.getElementById('videoSelectModal');
      if (modal) {
        modal.classList.add('show');
        modal.style.display = 'flex'; // Change from block to flex to center properly
        document.body.classList.add('overflow-hidden');
        
        // Load videos
        loadVideos();
      }
    });
  }
  
  // Setup refresh videos button
  const refreshVideosBtn = document.getElementById('refresh-videos-btn');
  if (refreshVideosBtn) {
    refreshVideosBtn.addEventListener('click', loadVideos);
  }
  
  // Setup search videos functionality
  const videoSearch = document.getElementById('video-search');
  if (videoSearch) {
    videoSearch.addEventListener('input', function() {
      const searchText = this.value.toLowerCase();
      const videoItems = document.querySelectorAll('.video-item');
      
      // For each item, check if the filename contains the search text
      videoItems.forEach(item => {
        const filename = item.querySelector('h4')?.textContent.toLowerCase() || '';
        if (searchText === '' || filename.includes(searchText)) {
          item.style.display = 'block';
        } else {
          item.style.display = 'none';
        }
      });
      
      // Show or hide the "no videos" message based on search results
      const visibleItems = Array.from(videoItems).filter(item => item.style.display !== 'none');
      const noVideos = document.getElementById('no-videos');
      
      if (visibleItems.length === 0 && searchText !== '' && noVideos) {
        noVideos.style.display = 'block';
        noVideos.innerHTML = `
          <div class="bg-dark-800 rounded-xl p-6 border border-dark-600">
            <i class="fas fa-search text-4xl mb-4 text-gray-500"></i>
            <p class="text-gray-300 text-lg">No videos matching "${searchText}"</p>
            <p class="text-gray-400 text-sm mt-2">Try a different search term</p>
          </div>
        `;
      } else if (noVideos && visibleItems.length > 0) {
        noVideos.style.display = 'none';
      }
    });
  }
  
  // Setup file input display
  const fileInput = document.getElementById('video-upload');
  if (fileInput) {
    fileInput.addEventListener('change', function() {
      const fileName = this.files[0]?.name || 'Select video file...';
      const fileNameElement = this.closest('.relative').querySelector('.file-name');
      if (fileNameElement) {
        fileNameElement.textContent = fileName;
      }
    });
  }
  
  // Setup auto-resizing textarea
  const chatInput = document.getElementById('chat-message-input');
  if (chatInput) {
    // Function to adjust height based on content
    function autoResizeTextarea() {
      chatInput.style.height = 'auto'; // Reset height to recalculate
      
      // Set a minimum height
      const minHeight = 48; // 3rem
      
      // Calculate new height based on scroll height (content)
      const newHeight = Math.min(chatInput.scrollHeight, 128); // 128px = 8rem (max-h-32)
      
      // Apply the new height, but not less than minimum
      chatInput.style.height = Math.max(newHeight, minHeight) + 'px';
    }
    
    // Initialize on load
    autoResizeTextarea();
    
    // Update on input
    chatInput.addEventListener('input', autoResizeTextarea);
    
    // Reset on submit
    document.querySelector('button[onclick="onChatMessageSubmit()"]').addEventListener('click', function() {
      setTimeout(() => {
        chatInput.style.height = '48px'; // Reset to minimum height after submit
      }, 10);
    });
  }
  
  // Modal functionality
  const modalTriggers = document.querySelectorAll('[data-modal-target]');
  const modalCloseButtons = document.querySelectorAll('[data-close-modal]');
  
  modalTriggers.forEach(trigger => {
    trigger.addEventListener('click', () => {
      const modalId = trigger.getAttribute('data-modal-target');
      const modal = document.getElementById(modalId);
      
      // Show modal with fade effect
      modal.classList.add('show');
      modal.style.display = 'flex'; // Use flex for centering
      document.body.classList.add('overflow-hidden');
    });
  });
  
  modalCloseButtons.forEach(button => {
    button.addEventListener('click', () => {
      const modal = button.closest('.modal');
      closeModal(modal);
    });
  });
  
  // Close modal when clicking on backdrop
  document.addEventListener('click', (e) => {
    const modals = document.querySelectorAll('.modal.show');
    modals.forEach(modal => {
      // If click is directly on the modal (the backdrop) but not on the dialog
      if (e.target === modal || e.target.classList.contains('fixed')) {
        if (!e.target.closest('.modal-dialog')) {
          closeModal(modal);
        }
      }
    });
  });
  
  function closeModal(modal) {
    if (!modal) return;
    
    modal.classList.remove('show');
    modal.classList.add('closing');
    
    setTimeout(() => {
      modal.style.display = 'none';
      modal.classList.remove('closing');
      document.body.classList.remove('overflow-hidden');
    }, 300);
  }
});

// Set the current time in the welcome message and connect websocket
document.addEventListener('DOMContentLoaded', function() {
  const welcomeTimeElement = document.getElementById('welcome-time');
  if (welcomeTimeElement) {
    const now = new Date();
    welcomeTimeElement.textContent = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  }
  
  // Connect to WebSocket server
  if (typeof connectWebsocket === 'function') {
    connectWebsocket(49000, handleServerMessage);
    console.log("Connected to WebSocket server");
  } else {
    console.error("WebSocket connection function not available");
    showToast("WebSocket connection not available", "error");
  }
  
  // Check if video is already loaded
  const videoElement = document.getElementById('surgery-video');
  if (videoElement && videoElement.src && videoElement.src !== window.location.href) {
    // Video is already loaded, enable the mic button
    enableMicButton();
    
    // Allow autoplay if the attribute is set in HTML
    // videoElement.pause();
    
    // Start automatic frame capture for annotations when video plays
    videoElement.addEventListener('play', () => {
      startAutoFrameCapture();
    }, { once: false });
  }
  
  // Set up video event listeners for frame capture if not already done
  if (videoElement) {
    // Clean up existing event listeners to avoid duplicates
    const existingListeners = videoElement._hasFrameCaptureListeners;
    if (!existingListeners) {
      videoElement.addEventListener('play', () => {
        console.log("Video playback started, enabling auto frame capture");
        startAutoFrameCapture();
      });
      
      videoElement.addEventListener('pause', () => {
        console.log("Video playback paused, disabling auto frame capture");
        stopAutoFrameCapture();
      });
      
      videoElement.addEventListener('ended', () => {
        console.log("Video playback ended, disabling auto frame capture");
        stopAutoFrameCapture();
      });
      
      // Mark that we've added these listeners
      videoElement._hasFrameCaptureListeners = true;
    }
    
    // Allow video autoplay if the attribute is set in HTML
    // videoElement.pause();
  }
});

// Handle messages from the server
function handleServerMessage(message) {
  console.log("Received message from server:", message);
  
  // Handle recognized text from audio
  if (message.recognized_text && message.asr_final) {
    addMessageToChat(message.recognized_text, 'user');
    // Send the message to be processed
    sendMessageToBackend(message.recognized_text);
  }
  
  // Handle AI responses
  if (message.message) {
    addMessageToChat(message.message, 'agent');
  }
  
  // Handle agent responses (new format)
  if (message.agent_response) {
    // Check if this is an annotation (contains marker text)
    if (message.agent_response.startsWith('Annotation:')) {
      // Add to annotations panel
      addAnnotation(message.agent_response);
      // Also add to chat as normal
      addMessageToChat(message.agent_response, 'agent');
    } 
    // Check if this is a note (or has is_note flag)
    else if (message.is_note || message.agent_response.toLowerCase().includes('note:')) {
      // Add to notes panel with user's original message if available
      addNote(message.agent_response, message.original_user_input || message.user_input || '');
      // Also add to chat
      addMessageToChat(message.agent_response, 'agent');
    }
    else {
      // Regular response - just add to chat
      addMessageToChat(message.agent_response, 'agent');
    }
    
    // Update phase if annotation includes phase info
    if (message.agent_response.includes("Phase '") || message.agent_response.includes("phase '")) {
      updatePhaseFromAnnotation(message.agent_response);
    }
  }
  
  // Handle video updates
  if (message.video_updated && message.video_src) {
    const videoElement = document.getElementById('surgery-video');
    if (videoElement) {
      // Pause current video first
      try {
        videoElement.pause();
      } catch (e) {
        console.warn("Could not pause video:", e);
      }
      
      // Set new source with autoplay
      videoElement.src = message.video_src;
      videoElement.load();
      videoElement.autoplay = true;
      
      // Reset UI elements for new video
      const phaseElement = document.getElementById('current-phase');
      if (phaseElement) {
        phaseElement.textContent = 'Undefined';
      }
      
      // Clear annotations for new video
      const annotationsContainer = document.getElementById('annotations-container');
      if (annotationsContainer) {
        annotationsContainer.innerHTML = `
          <div class="text-center text-gray-400 p-5">
            <i class="fas fa-tag fa-3x mb-3"></i>
            <p>No annotations available yet. Annotations will appear here as they are generated.</p>
          </div>
        `;
      }
      
      // Reset annotation count
      const annotationCount = document.querySelector('.annotation-count');
      if (annotationCount) {
        annotationCount.textContent = '0';
      }
    }
  }
  
  // Handle request for frame
  if (message.request_frame) {
    sendFrameWithText(message.recognized_text);
  }
}

// Function to add annotation to the annotations panel
function addAnnotation(annotationText) {
  const annotationsContainer = document.getElementById('annotations-container');
  const noAnnotationsMsg = annotationsContainer.querySelector('.text-center');
  
  // Remove "no annotations" message if it exists
  if (noAnnotationsMsg) {
    noAnnotationsMsg.remove();
  }
  
  // Create annotation element
  const annotationElement = document.createElement('div');
  annotationElement.className = 'bg-dark-800 rounded-lg p-3 border border-dark-700 mb-3';
  
  const now = new Date();
  const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  
  // Parse the annotation text to extract phase, tools, anatomy
  let phaseText = '';
  let toolsText = '';
  let anatomyText = '';
  
  if (annotationText.includes('Phase')) {
    const phaseMatch = annotationText.match(/Phase [\'"]([^'"]+)[\'"]/) || 
                       annotationText.match(/phase [\'"]([^'"]+)[\'"]/) || 
                       annotationText.match(/Phase: ([^|]+)/);
    if (phaseMatch) {
      phaseText = phaseMatch[1];
    }
  }
  
  if (annotationText.includes('Tools:')) {
    const toolsMatch = annotationText.match(/Tools: ([^|]+)/);
    if (toolsMatch) {
      toolsText = toolsMatch[1].trim();
    }
  }
  
  if (annotationText.includes('Anatomy:')) {
    const anatomyMatch = annotationText.match(/Anatomy: ([^$]+)/);
    if (anatomyMatch) {
      anatomyText = anatomyMatch[1].trim();
    }
  }
  
  annotationElement.innerHTML = `
    <div class="flex justify-between items-start mb-2">
      <div>
        <h3 class="text-lg font-semibold text-success-400">${phaseText || 'Annotation'}</h3>
        <div class="flex flex-wrap gap-1 mt-1">
          ${toolsText ? `<span class="badge bg-dark-700 text-primary-300 px-2 py-0.5"><i class="fas fa-tools mr-1"></i>${toolsText}</span>` : ''}
          ${anatomyText ? `<span class="badge bg-dark-700 text-yellow-300 px-2 py-0.5"><i class="fas fa-heart mr-1"></i>${anatomyText}</span>` : ''}
        </div>
      </div>
      <span class="text-xs text-gray-400">${timeString}</span>
    </div>
    <p class="text-sm text-gray-300">${annotationText}</p>
  `;
  
  // Add to container
  annotationsContainer.prepend(annotationElement);
  
  // Update the count
  const annotationCount = document.querySelector('.annotation-count');
  if (annotationCount) {
    annotationCount.textContent = parseInt(annotationCount.textContent || '0') + 1;
  }
}

// Function to add note to the notes panel
function addNote(noteText, userMessage) {
  const notesContainer = document.getElementById('notes-container');
  const noNotesMsg = notesContainer.querySelector('.text-center');
  
  // Check if this note already exists to prevent duplicates
  const existingNotes = notesContainer.querySelectorAll('.note-content');
  for (let i = 0; i < existingNotes.length; i++) {
    if (existingNotes[i].textContent.trim() === noteText.trim()) {
      console.log("Duplicate note detected, not adding again");
      return; // Exit if duplicate found
    }
  }
  
  // Remove "no notes" message if it exists
  if (noNotesMsg) {
    noNotesMsg.remove();
  }
  
  // Create note element with modern styling
  const noteElement = document.createElement('div');
  noteElement.className = 'bg-dark-800 rounded-lg p-3 border border-dark-700 mb-3 hover:shadow-md transition-all duration-200';
  
  const now = new Date();
  const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const dateString = now.toLocaleDateString([], { month: 'short', day: 'numeric' });
  
  // Get current video time if available
  const currentVideo = document.getElementById('surgery-video');
  
  // Extract title and content with improved parsing
  let title = 'Note';
  let content = noteText;
  let category = 'General';
  
  // Process AI response for note content
  if (noteText.toLowerCase().includes('note:')) {
    // Case: AI returned a formatted note
    const noteMatch = noteText.match(/note:([^$]+)/i);
    if (noteMatch && noteMatch[1]) {
      content = noteMatch[1].trim();
      
      // Try to extract a title from the first sentence
      const firstSentence = content.split(/[.!?]/, 1)[0].trim();
      if (firstSentence && firstSentence.length < 50) {
        title = firstSentence;
        // Remove the title from the content if it's at the beginning
        content = content.replace(firstSentence, '').trim();
        if (content.startsWith('.') || content.startsWith('!') || content.startsWith('?')) {
          content = content.substring(1).trim();
        }
      }
    }
  } 
  
  // Handle user's "take a note" command - prioritize this over AI response
  if (userMessage && userMessage.toLowerCase().includes('take a note')) {
    // Extract the note content from the user's message
    const noteMatch = userMessage.match(/take a note(?: about| on| regarding)? (.+)/i);
    if (noteMatch && noteMatch[1]) {
      // Use the user's topic as the title
      title = noteMatch[1].trim();
      
      // Extract meaningful content from the AI response
      if (noteText && !noteText.toLowerCase().includes('note recorded') && !noteText.toLowerCase().includes('timestamp=')) {
        // If AI provided actual useful content, use it
        content = noteText;
      } else {
        // If AI didn't provide useful content, create a meaningful note based on user's request
        const currentVideo = document.getElementById('surgery-video');
        const currentTime = currentVideo ? formatTime(currentVideo.currentTime) : timeString;
        
        // Generate useful note content based on topic
        if (title.toLowerCase().includes('bleed')) {
          content = `Bleeding observed at ${currentTime}. Patient requires attention to the affected area.`;
        } else if (title.toLowerCase().includes('tool') || title.toLowerCase().includes('instrument')) {
          content = `Tool usage noted at ${currentTime}. Proper technique observed.`;
        } else if (title.toLowerCase().includes('anatomy') || title.toLowerCase().includes('structure')) {
          content = `Anatomical structure identified at ${currentTime}. Clear visualization achieved.`;
        } else if (title.toLowerCase().includes('phase') || title.toLowerCase().includes('stage')) {
          content = `Procedure phase change at ${currentTime}. Moving to next steps.`;
        } else {
          content = `Observation at ${currentTime}: ${title} noted during procedure.`;
        }
      }
    }
  }
  
  // Determine category based on content keywords
  if (content.toLowerCase().includes('bleed') || content.toLowerCase().includes('blood')) {
    category = 'Bleeding';
  } else if (content.toLowerCase().includes('tool') || content.toLowerCase().includes('instrument')) {
    category = 'Tools';
  } else if (content.toLowerCase().includes('anatomy') || content.toLowerCase().includes('organ')) {
    category = 'Anatomy';
  } else if (content.toLowerCase().includes('procedure') || content.toLowerCase().includes('technique')) {
    category = 'Procedure';
  }
  
  noteElement.innerHTML = `
    <div class="flex justify-between items-start mb-2">
      <div class="flex-1">
        <div class="flex items-center">
          <h3 class="text-sm font-semibold text-primary-400 truncate">${title}</h3>
          <span class="ml-2 text-[10px] font-medium bg-primary-900/50 text-primary-300 px-1.5 py-0.5 rounded-full">${category}</span>
        </div>
        <div class="text-xs text-gray-400 mt-0.5 mb-1.5">${timeString} · Video: ${currentVideo ? formatTime(currentVideo.currentTime) : 'N/A'}</div>
      </div>
      <div class="flex ml-2 mt-1 space-x-1">
        <button class="edit-note-btn w-6 h-6 flex items-center justify-center text-xs text-gray-400 hover:text-primary-300 transition-colors rounded-full bg-dark-800 hover:bg-dark-700">
          <i class="fas fa-edit"></i>
        </button>
        <button class="delete-note-btn w-6 h-6 flex items-center justify-center text-xs text-gray-400 hover:text-red-400 transition-colors rounded-full bg-dark-800 hover:bg-dark-700">
          <i class="fas fa-trash-alt"></i>
        </button>
      </div>
    </div>
    <div class="text-xs text-gray-300 note-content leading-relaxed border-l-2 border-primary-800/30 pl-2 mb-1 max-h-24 overflow-y-auto">
      ${content}
    </div>
  `;
  
  // Add to container
  notesContainer.prepend(noteElement);
  
  // Add event listeners for the buttons
  const editBtn = noteElement.querySelector('.edit-note-btn');
  if (editBtn) {
    editBtn.addEventListener('click', function() {
      editNote(noteElement, title, content);
    });
  }
  
  const deleteBtn = noteElement.querySelector('.delete-note-btn');
  if (deleteBtn) {
    deleteBtn.addEventListener('click', function() {
      deleteNote(noteElement);
    });
  }
  
  // Update the count
  const notesCount = document.querySelector('.notes-count');
  if (notesCount) {
    notesCount.textContent = parseInt(notesCount.textContent || '0') + 1;
  }
}

// Function to edit an existing note
function editNote(noteElement, title, content) {
  // Get the note modal elements
  const modal = document.getElementById('addNoteModal');
  const modalTitle = modal.querySelector('.modal-title');
  const titleInput = document.getElementById('note-title');
  const contentInput = document.getElementById('note-content');
  const saveButton = document.getElementById('save-note-btn');
  
  // Change modal title to indicate editing
  if (modalTitle) {
    modalTitle.innerHTML = '<i class="fas fa-edit text-primary-400 mr-2"></i> Edit Note';
  }
  
  // Pre-fill the form with existing values
  if (titleInput) titleInput.value = title;
  if (contentInput) contentInput.value = content;
  
  // Temporarily store the note element to edit
  saveButton.setAttribute('data-editing', 'true');
  saveButton.setAttribute('data-note-id', Date.now().toString()); // Use timestamp as a makeshift ID
  noteElement.id = saveButton.getAttribute('data-note-id');
  
  // Change save button text
  saveButton.innerHTML = '<i class="fas fa-save mr-1.5"></i> Update Note';
  
  // Update the save handler
  saveButton.onclick = function() {
    // Get the updated values
    const newTitle = titleInput.value.trim();
    const newContent = contentInput.value.trim();
    const message = document.getElementById('note-message').value.trim();
    
    if (!newTitle || !newContent) {
      showToast('Please enter a title and content for your note', 'error');
      return;
    }
    
    // Update the note HTML
    const noteToUpdate = document.getElementById(this.getAttribute('data-note-id'));
    if (noteToUpdate) {
      const titleEl = noteToUpdate.querySelector('h3');
      const contentEl = noteToUpdate.querySelector('.note-content');
      
      if (titleEl) titleEl.textContent = newTitle;
      if (contentEl) contentEl.innerHTML = newContent;
      
      // Close modal and clean up
      closeModal(modal);
      resetNoteForm();
      
      showToast('Note updated successfully', 'success');
      
      // Send any message to chat if provided
      if (message) {
        addMessageToChat(message, 'user');
        sendMessageToBackend(message);
      }
    }
  };
  
  // Open the modal
  modal.classList.add('show');
  modal.style.display = 'flex';
  document.body.classList.add('overflow-hidden');
}

// Function to delete a note
function deleteNote(noteElement) {
  // Ask for confirmation
  if (confirm('Are you sure you want to delete this note?')) {
    // Remove the note element
    noteElement.classList.add('opacity-0', 'scale-95');
    noteElement.style.transition = 'all 0.3s ease-in-out';
    
    // Add a slight delay before actually removing the element
    setTimeout(() => {
      noteElement.remove();
      
      // Update the count
      const notesCount = document.querySelector('.notes-count');
      if (notesCount) {
        const currentCount = parseInt(notesCount.textContent || '0');
        notesCount.textContent = Math.max(0, currentCount - 1);
      }
      
      // Check if there are no more notes and show the empty message
      const notesContainer = document.getElementById('notes-container');
      if (notesContainer && notesContainer.children.length === 0) {
        notesContainer.innerHTML = `
          <div class="text-center text-gray-400 p-4">
            <div class="flex items-center justify-center mb-2">
              <i class="fas fa-sticky-note text-xl mr-2 text-primary-700 opacity-60"></i>
              <span class="text-sm">No notes available</span>
            </div>
            <p class="text-xs">Ask the assistant to "take a note about..." something you observe</p>
          </div>
        `;
      }
      
      showToast('Note deleted successfully', 'success');
    }, 300);
  }
}

// Function to reset the note form
function resetNoteForm() {
  const modal = document.getElementById('addNoteModal');
  const modalTitle = modal.querySelector('.modal-title');
  const titleInput = document.getElementById('note-title');
  const contentInput = document.getElementById('note-content');
  const messageInput = document.getElementById('note-message');
  const saveButton = document.getElementById('save-note-btn');
  
  // Reset title
  if (modalTitle) {
    modalTitle.innerHTML = '<i class="fas fa-sticky-note text-primary-400 mr-2"></i> Add Note';
  }
  
  // Clear form inputs
  if (titleInput) titleInput.value = '';
  if (contentInput) contentInput.value = '';
  if (messageInput) messageInput.value = '';
  
  // Reset save button
  saveButton.removeAttribute('data-editing');
  saveButton.removeAttribute('data-note-id');
  saveButton.innerHTML = '<i class="fas fa-save mr-1.5"></i> Save Note';
  
  // Reset the onclick handler
  saveButton.onclick = saveManualNote;
  
  // Hide image preview if any
  const previewContainer = document.getElementById('note-image-preview-container');
  if (previewContainer) previewContainer.classList.add('hidden');
}

// Update the phase tag under the video
function updatePhaseFromAnnotation(annotationText) {
  const phaseElement = document.getElementById('current-phase');
  if (!phaseElement) return;
  
  const phaseMatch = annotationText.match(/Phase [\'"]([^'"]+)[\'"]/) || 
                     annotationText.match(/phase [\'"]([^'"]+)[\'"]/) || 
                     annotationText.match(/Phase: ([^|]+)/);
  
  if (phaseMatch) {
    const phaseName = phaseMatch[1].trim();
    phaseElement.textContent = phaseName;
    
    // Add animation to highlight the change
    phaseElement.style.transition = 'all 0.3s ease';
    phaseElement.style.backgroundColor = '#16a34a'; // success-600
    phaseElement.style.color = 'white';
    
    setTimeout(() => {
      phaseElement.style.backgroundColor = '';
      phaseElement.style.color = '';
    }, 1500);
  }
}

// Format time in MM:SS format (shared utility function)
function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${minutes}:${secs < 10 ? '0' : ''}${secs}`;
}

// Video time tracking
document.addEventListener('DOMContentLoaded', function() {
  const video = document.getElementById('surgery-video');
  const currentTimeDisplay = document.getElementById('video-current-time');
  const durationDisplay = document.getElementById('video-duration');
    
  if (video && currentTimeDisplay && durationDisplay) {
    // Update time displays when metadata is loaded
    video.addEventListener('loadedmetadata', function() {
      durationDisplay.textContent = formatTime(video.duration);
    });
    
    // Update current time during playback
    video.addEventListener('timeupdate', function() {
      currentTimeDisplay.textContent = formatTime(video.currentTime);
    });
  }
});

// Fullscreen functionality for video
function toggleFullscreen() {
  const videoContainer = document.getElementById('video-container');
  
  if (!document.fullscreenElement) {
    if (videoContainer.requestFullscreen) {
      videoContainer.requestFullscreen();
    } else if (videoContainer.webkitRequestFullscreen) { /* Safari */
      videoContainer.webkitRequestFullscreen();
    } else if (videoContainer.msRequestFullscreen) { /* IE11 */
      videoContainer.msRequestFullscreen();
    }
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    } else if (document.webkitExitFullscreen) { /* Safari */
      document.webkitExitFullscreen();
    } else if (document.msExitFullscreen) { /* IE11 */
      document.msExitFullscreen();
    }
  }
}

// Custom toast notifications
function showToast(message, type = 'info', duration = 4000) {
  // Create toast container if it doesn't exist
  let toastContainer = document.querySelector('.toast-container');
  if (!toastContainer) {
    toastContainer = document.createElement('div');
    toastContainer.className = 'toast-container';
    // Use inline styles to ensure positioning works
    Object.assign(toastContainer.style, {
      position: 'fixed',
      top: '1rem',
      right: '1rem',
      zIndex: '9999',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'flex-end',
      gap: '0.5rem',
      maxWidth: '20rem'
    });
    document.body.appendChild(toastContainer);
  }
  
  // Create toast element
  const toast = document.createElement('div');
  toast.className = `custom-toast ${type}`;
  
  // Apply base toast styles inline to ensure they're applied
  Object.assign(toast.style, {
    display: 'flex',
    alignItems: 'center',
    padding: '0.75rem 1rem',
    borderRadius: '0.5rem',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
    transform: 'translateX(100%)',
    opacity: '0',
    transition: 'all 0.3s ease',
    color: 'white',
    width: '100%',
    marginBottom: '0.5rem',
    borderLeft: '4px solid'
  });
  
  // Add appropriate icon and styling based on type
  let icon = 'info-circle';
  let bgColor, borderColor;
  
  if (type === 'success') {
    icon = 'check-circle';
    bgColor = 'linear-gradient(to right, #059669, #047857)'; // green-600 to green-700
    borderColor = '#34d399'; // green-400
  } else if (type === 'error') {
    icon = 'exclamation-triangle';
    bgColor = 'linear-gradient(to right, #dc2626, #b91c1c)'; // red-600 to red-700
    borderColor = '#f87171'; // red-400
  } else {
    // info default
    bgColor = 'linear-gradient(to right, #0369a1, #075985)'; // primary-700 to primary-800
    borderColor = '#38bdf8'; // primary-400
  }
  
  toast.style.background = bgColor;
  toast.style.borderLeftColor = borderColor;
  
  // Add close button and improved styling
  toast.innerHTML = `
    <i class="fas fa-${icon}" style="margin-right: 0.75rem; font-size: 1.1rem;"></i>
    <span style="flex-grow: 1; font-size: 0.9rem;">${message}</span>
    <button style="margin-left: 0.5rem; opacity: 0.8; cursor: pointer; background: none; border: none; color: white;" 
            onmouseover="this.style.opacity='1'" 
            onmouseout="this.style.opacity='0.8'" 
            onclick="this.parentNode.remove()">
      <i class="fas fa-times"></i>
    </button>
  `;
  
  // Add to container
  toastContainer.appendChild(toast);
  
  // Trigger animation after a small delay
  setTimeout(() => {
    toast.style.transform = 'translateX(0)';
    toast.style.opacity = '1';
  }, 10);
  
  // Remove after duration
  const timeoutId = setTimeout(() => {
    toast.style.transform = 'translateX(100%)';
    toast.style.opacity = '0';
    setTimeout(() => {
      if (toast.parentNode) {
        toast.remove();
      }
    }, 300);
  }, duration);
  
  // Cancel timeout when manually closed
  toast.querySelector('button').addEventListener('click', () => {
    clearTimeout(timeoutId);
  });
}

// Chat functions
function onChatMessageKey(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    onChatMessageSubmit();
  }
}

function onChatMessageSubmit() {
  const inputElement = document.getElementById('chat-message-input');
  const message = inputElement.value.trim();
  
  if (message) {
    // Clear input
    inputElement.value = '';
    
    // Add to chat history
    addMessageToChat(message, 'user');
    
    // Send to backend with current video frame
    sendMessageToBackend(message);
  }
}

// Function to send a message to the backend with a video frame
function sendMessageToBackend(message) {
  try {
    // Prepare payload with message
    const payload = {
      user_input: message,
      original_user_input: message // Store original message for note processing
    };
    
    // First try to get a new frame by capturing the current video
    let frameData = captureVideoFrame();
    
    // If we couldn't get a new frame, try the previously stored frame
    if (!frameData) {
      console.warn("Could not capture current frame, trying to use last captured frame");
      frameData = sessionStorage.getItem('lastCapturedFrame');
    }
    
    // If we still don't have a frame but have a video loaded, try seeking and capturing
    if (!frameData) {
      const videoElement = document.getElementById('surgery-video');
      if (videoElement && videoElement.readyState >= 2) {
        // Try to seek to first frame to ensure we can capture something
        try {
          console.warn("Attempting frame capture by seeking to beginning of video");
          
          // Store current playback state and position
          const wasPlaying = !videoElement.paused;
          const currentTime = videoElement.currentTime;
          
          // Pause if playing
          if (wasPlaying) {
            videoElement.pause();
          }
          
          // Seek to a small offset (0.1 second) to ensure a frame is available
          videoElement.currentTime = 0.1;
          
          // Wait a tiny bit for the frame to load
          setTimeout(() => {
            // Try to capture again
            frameData = captureVideoFrame();
            
            if (frameData) {
              console.log("Successfully captured frame after seeking");
              sessionStorage.setItem('lastCapturedFrame', frameData);
              
              // Send the message with the new frame
              completeMessageSend(message, frameData);
            } else {
              // Still couldn't get a frame, use placeholders
              fallbackFrameCapture(message);
            }
            
            // Restore playback state
            try {
              videoElement.currentTime = currentTime;
              if (wasPlaying) {
                videoElement.play().catch(e => console.warn("Could not resume playback:", e));
              }
            } catch (e) {
              console.warn("Error restoring video state:", e);
            }
          }, 50);
          
          // Return early as we'll send the message in the callback
          return;
        } catch (e) {
          console.warn("Error while trying to seek for frame capture:", e);
        }
      }
      
      // If we're still here, try placeholder
      fallbackFrameCapture(message);
      return;
    }
    
    // If we got here with a frame, send it
    completeMessageSend(message, frameData);
  } catch (err) {
    console.error("Error sending message to backend:", err);
    showToast("Error sending message: " + err.message, "error");
    
    // Try to send just the message without a frame
    if (typeof sendJSON === 'function') {
      sendJSON({
        user_input: message,
        original_user_input: message
      });
      console.log("Sent message WITHOUT frame data due to error");
    }
  }
}

// Function to create a placeholder image frame
function createPlaceholderFrame() {
  // Check if we already have a placeholder frame
  let frameData = sessionStorage.getItem('placeholderFrame');
  if (frameData) {
    console.log("Using existing placeholder frame");
    return frameData;
  }
  
  try {
    // Create a placeholder canvas with text
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 360;
    const ctx = canvas.getContext('2d');
    
    // Fill with dark background
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Add text indicating this is a placeholder
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Surgical Agentic Framework Demo', canvas.width/2, canvas.height/2 - 20);
    
    ctx.font = '16px Arial';
    ctx.fillText('A frame will be captured when a video is playing', canvas.width/2, canvas.height/2 + 20);
    
    // Convert to data URL and store it
    frameData = canvas.toDataURL('image/jpeg', 0.8);
    sessionStorage.setItem('placeholderFrame', frameData);
    console.log("Created new placeholder frame");
    return frameData;
  } catch (err) {
    console.error("Error creating placeholder frame:", err);
    return null;
  }
}

// Create placeholder frame when all other methods fail
function fallbackFrameCapture(message) {
  console.warn("No previously captured frame available, using fallback");
  let frameData = createPlaceholderFrame();
  
  // If we still don't have a frame, create a message-specific one
  if (!frameData) {
    try {
      // Create a placeholder canvas with text
      const canvas = document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 360;
      const ctx = canvas.getContext('2d');
      
      // Fill with dark background
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Add text indicating no frame is available
      ctx.fillStyle = '#ffffff';
      ctx.font = '20px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('No video frame available', canvas.width/2, canvas.height/2);
      ctx.font = '16px Arial';
      ctx.fillText('Question: "' + message.substring(0, 30) + (message.length > 30 ? '...' : '') + '"', canvas.width/2, canvas.height/2 + 30);
      
      // Convert to data URL
      frameData = canvas.toDataURL('image/jpeg', 0.8);
      
      // Store this placeholder for potential future use
      sessionStorage.setItem('placeholderFrame', frameData);
    } catch (canvasErr) {
      console.error("Error creating placeholder frame:", canvasErr);
    }
  }
  
  completeMessageSend(message, frameData);
}

// Final step to send message with whatever frame we have
function completeMessageSend(message, frameData) {
  // Prepare payload with message
  const payload = {
    user_input: message,
    original_user_input: message
  };
  
  // Always try to provide some frame data, even if it's a placeholder
  if (!frameData) {
    console.warn("No frame data provided, creating placeholder");
    frameData = createPlaceholderFrame();
  }
  
  // Add frame data to payload
  if (frameData) {
    payload.frame_data = frameData;
    console.log("Frame data added to message payload");
  } else {
    // This should almost never happen since we generate placeholders
    console.warn("Failed to create any frame data to send with message");
  }
  
  // Send payload to server
  if (typeof sendJSON === 'function') {
    sendJSON(payload);
    console.log("Message sent to backend" + (frameData ? " with frame data" : " WITHOUT frame data"));
  } else {
    console.error("sendJSON function not available");
    showToast("Unable to send message to server", "error");
  }
}

// Function to send frame data with text
function sendFrameWithText(text) {
  try {
    // Delegate to the common send function with force capture enabled
    sendTextWithMaxCapture(text);
  } catch (err) {
    console.error("Error sending frame with text:", err);
    
    // Fallback - just send the text without a frame
    if (typeof sendJSON === 'function') {
      sendJSON({
        user_input: text,
        asr_final: true
      });
      console.log("Text sent without frame (error recovery): " + text);
    }
  }
}

// Enhanced function that tries harder to get a frame from the video
function sendTextWithMaxCapture(text) {
  // Try to get a frame directly
  let frameData = captureVideoFrame();
  
  // If that worked, use it
  if (frameData) {
    sessionStorage.setItem('lastCapturedFrame', frameData);
    completeFrameWithTextSend(text, frameData);
    return;
  }
  
  // Otherwise try getting the last stored frame
  frameData = sessionStorage.getItem('lastCapturedFrame');
  if (frameData) {
    completeFrameWithTextSend(text, frameData);
    return;
  }
  
  // If we still don't have a frame but video is loaded, try seek and capture
  const videoElement = document.getElementById('surgery-video');
  if (videoElement && videoElement.readyState >= 2) {
    // Try to force a frame capture
    const wasPlaying = !videoElement.paused;
    const currentTime = videoElement.currentTime;
    
    // Pause playback
    if (wasPlaying) {
      videoElement.pause();
    }
    
    try {
      // Seek to start if needed to ensure we get a frame
      if (videoElement.currentTime > 10) {
        // If we're far into the video, seek to 0.1s
        videoElement.currentTime = 0.1;
      } else {
        // Otherwise just seek a tiny bit forward to force frame update
        videoElement.currentTime = Math.max(0.1, videoElement.currentTime + 0.1);
      }
      
      // Wait a brief moment for the frame to load
      setTimeout(() => {
        // Try to capture again
        frameData = captureVideoFrame();
        
        if (frameData) {
          // Success! Store and send
          sessionStorage.setItem('lastCapturedFrame', frameData);
          completeFrameWithTextSend(text, frameData);
        } else {
          // Still no frame - use placeholder
          createPlaceholderFrameWithText(text);
        }
        
        // Restore video state
        try {
          videoElement.currentTime = currentTime;
          if (wasPlaying) {
            videoElement.play().catch(e => {});
          }
        } catch (e) {}
      }, 100);
      
      return; // Will continue in the timeout
    } catch (e) {
      console.warn("Error during forced frame capture:", e);
      // Continue to placeholder
    }
  }
  
  // If all else failed, use a placeholder
  createPlaceholderFrameWithText(text);
}

// Create a placeholder frame with the specified text
function createPlaceholderFrameWithText(text) {
  // Check if we already have a placeholder
  let frameData = sessionStorage.getItem('placeholderFrame');
  
  // If not, create one
  if (!frameData) {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 360;
      const ctx = canvas.getContext('2d');
      
      // Dark background
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      
      // Add helpful text
      ctx.fillStyle = '#ffffff';
      ctx.font = '20px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('No video frame available', canvas.width/2, canvas.height/2);
      ctx.font = '16px Arial';
      ctx.fillText('Voice command: "' + text.substring(0, 30) + (text.length > 30 ? '...' : '') + '"', canvas.width/2, canvas.height/2 + 30);
      
      // Convert to data URL
      frameData = canvas.toDataURL('image/jpeg', 0.8);
      
      // Store for future use
      sessionStorage.setItem('placeholderFrame', frameData);
    } catch (canvasErr) {
      console.error("Error creating placeholder frame:", canvasErr);
    }
  }
  
  // Send whatever we have
  completeFrameWithTextSend(text, frameData);
}

// Complete the send operation with whatever frame we've managed to get
function completeFrameWithTextSend(text, frameData) {
  // Prepare payload with required fields
  const payload = {
    user_input: text,
    asr_final: true
  };
  
  // Add frame data if we have it
  if (frameData) {
    payload.frame_data = frameData;
  }
  
  // Send to server
  if (typeof sendJSON === 'function') {
    sendJSON(payload);
    console.log("Voice input sent" + (frameData ? " with frame" : " WITHOUT frame") + ": " + text);
    
    // Show a warning to the user if no frame was available
    if (!frameData) {
      showToast("No video frame available - AI response may be limited", "warning");
    }
  } else {
    console.error("sendJSON function not available");
    showToast("Unable to send message", "error");
  }
}

function addMessageToChat(message, sender = 'user') {
  const chatHistoryContainer = document.getElementById('chat-history-container');
  
  // Check if this message already exists (to prevent duplicates)
  const lastMessage = chatHistoryContainer.lastElementChild;
  if (lastMessage && 
      lastMessage.classList.contains(sender === 'user' ? 'user-message' : 'agent-message') &&
      lastMessage.querySelector('.message-content').textContent === message) {
    console.log("Duplicate message detected, not adding again");
    return;
  }
  
  const messageDiv = document.createElement('div');
  const now = new Date();
  const timeString = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  
  if (sender === 'user') {
    messageDiv.className = 'user-message';
    messageDiv.innerHTML = `
      <div class="flex items-center justify-between mb-0 -mt-1 -mx-1">
        <span class="flex items-center">
          <span class="avatar-icon bg-primary-700">
            <i class="fas fa-user"></i>
          </span>
          <span class="text-[11px] text-white/90">You</span>
        </span>
        <span class="text-[10px] text-primary-200/80">${timeString}</span>
      </div>
      <div class="message-content">
        ${message}
      </div>
    `;
  } else {
    messageDiv.className = 'agent-message';
    messageDiv.innerHTML = `
      <div class="flex items-center justify-between mb-0 -mt-1 -mx-1">
        <span class="flex items-center">
          <span class="avatar-icon bg-success-600">
            <i class="fas fa-robot"></i>
          </span>
          <span class="text-[11px] text-success-300/90">AI Assistant</span>
        </span>
        <span class="text-[10px] text-gray-400/80">${timeString}</span>
      </div>
      <div class="message-content">
        ${message}
      </div>
    `;
    
    // Add entry animation
    messageDiv.style.opacity = '0';
    messageDiv.style.transform = 'translateY(10px)';
    
    setTimeout(() => {
      messageDiv.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
      messageDiv.style.opacity = '1';
      messageDiv.style.transform = 'translateY(0)';
    }, 10);
  }
  
  chatHistoryContainer.appendChild(messageDiv);
  chatHistoryContainer.scrollTop = chatHistoryContainer.scrollHeight;
}

function onChatHistoryReset() {
  const chatHistoryContainer = document.getElementById('chat-history-container');
  
  // Remove all messages except the welcome message
  while (chatHistoryContainer.childNodes.length > 1) {
    chatHistoryContainer.removeChild(chatHistoryContainer.lastChild);
  }
}

// Functions for mic control
function toggleMic() {
  const micBtn = document.getElementById('mic-btn');
  
  // Check if button is disabled
  if (micBtn.disabled) {
    showToast('Please load a video before using the microphone', 'error');
    return;
  }
  
  const isRecording = micBtn.classList.contains('recording');
  
  if (isRecording) {
    // Stop recording - Return to blue
    micBtn.classList.remove('recording');
    micBtn.classList.remove('from-red-600', 'to-red-700');
    micBtn.classList.remove('hover:from-red-500', 'hover:to-red-600');
    micBtn.classList.add('from-primary-600', 'to-primary-700');
    micBtn.classList.add('hover:from-primary-500', 'hover:to-primary-600');
    micBtn.innerHTML = '<i class="fas fa-microphone mr-2"></i> <span>Start Mic</span>';
    
    // Stop the recording
    stopRecording();
  } else {
    // Start recording - Make button red
    micBtn.classList.add('recording');
    micBtn.classList.remove('from-primary-600', 'to-primary-700');
    micBtn.classList.remove('hover:from-primary-500', 'hover:to-primary-600');
    micBtn.classList.add('from-red-600', 'to-red-700');
    micBtn.classList.add('hover:from-red-500', 'hover:to-red-600');
    micBtn.innerHTML = '<i class="fas fa-stop-circle mr-2"></i> <span>Stop Mic</span>';
    
    // Start the recording
    startRecording();
  }
}

// Function to enable the mic button when a video is loaded
function enableMicButton() {
  const micBtn = document.getElementById('mic-btn');
  if (micBtn) {
    micBtn.disabled = false;
    micBtn.classList.remove('opacity-50', 'cursor-not-allowed');
    console.log('Microphone button enabled');
  }
}

// Connect to the audio.js functions for recording
function startRecording() {
  try {
    // Capture a frame at the start of recording to ensure we have one
    const frameData = captureVideoFrame();
    if (frameData) {
      sessionStorage.setItem('lastCapturedFrame', frameData);
      console.log("Captured initial frame for voice recording");
    }
    
    if (typeof startAudio === 'function') {
      startAudio();
      showToast('Recording started', 'info');
    } else {
      console.error('startAudio function not found!');
      showToast('Error starting recording - audio.js not loaded correctly', 'error');
    }
  } catch (err) {
    console.error('Error starting recording:', err);
    showToast('Failed to start recording: ' + err.message, 'error');
  }
}

function stopRecording() {
  try {
    // Capture a final frame before stopping
    const frameData = captureVideoFrame();
    if (frameData) {
      sessionStorage.setItem('lastCapturedFrame', frameData);
      console.log("Captured final frame for voice recording");
    }
    
    if (typeof stopAudio === 'function') {
      stopAudio();
      showToast('Processing your voice input...', 'info');
    } else {
      console.error('stopAudio function not found!');
      showToast('Error stopping recording - audio.js not loaded correctly', 'error');
    }
  } catch (err) {
    console.error('Error stopping recording:', err);
    showToast('Failed to stop recording: ' + err.message, 'error');
  }
}

// Image capture function
function captureAndStoreFrame() {
  try {
    const frameData = captureVideoFrame();
    if (frameData) {
      // Store the captured frame in sessionStorage
      sessionStorage.setItem('lastCapturedFrame', frameData);
      showToast('Frame captured successfully!', 'success');
    } else {
      showToast('Failed to capture frame - video not playing', 'error');
    }
  } catch (err) {
    console.error('Error capturing frame:', err);
    showToast('Error capturing frame: ' + err.message, 'error');
  }
}

// Auto frame capture for annotations
let frameCapture = null;
const FRAME_CAPTURE_INTERVAL = 10000; // 10 seconds

function startAutoFrameCapture() {
  // Clear any existing interval
  stopAutoFrameCapture();
  
  // Force a capture now to ensure we have a frame, even if the video is paused
  let initialFrame = captureVideoFrame();
  
  // If we couldn't capture, but a video is loaded, try to seek to the first frame
  if (!initialFrame) {
    const videoElement = document.getElementById('surgery-video');
    if (videoElement && videoElement.readyState >= 2) {
      // Try to seek to first frame to ensure we can capture something
      try {
        // Store current playback state
        const wasPlaying = !videoElement.paused;
        
        // Pause if playing
        if (wasPlaying) {
          videoElement.pause();
        }
        
        // Seek to a small offset (0.1 second) to ensure a frame is available
        videoElement.currentTime = 0.1;
        
        // Try to capture again
        console.log("Attempting to capture after seeking to first frame");
        initialFrame = captureVideoFrame();
        
        // Resume playback if it was playing before
        if (wasPlaying) {
          videoElement.play().catch(e => console.warn("Could not resume playback:", e));
        }
      } catch (e) {
        console.warn("Error while trying to seek for frame capture:", e);
      }
    }
  }
  
  // If we now have a frame, store and send it
  if (initialFrame) {
    // Store the successfully captured frame
    sessionStorage.setItem('lastCapturedFrame', initialFrame);
    console.log("Initial frame captured and stored successfully");
    
    // Send to server for annotation
    if (typeof sendJSON === 'function') {
      sendJSON({
        auto_frame: true,
        frame_data: initialFrame
      });
      console.log("Initial frame sent for annotation");
    }
  } else {
    console.warn("Failed to capture initial frame - ChatBot responses may be limited");
    showToast("Could not capture video frame - AI responses may be limited", "warning");
  }
  
  // Start regular interval for frame capture
  frameCapture = setInterval(() => {
    // Try to capture current frame
    const frameData = captureVideoFrame();
    
    if (frameData) {
      // Store the frame for future use
      sessionStorage.setItem('lastCapturedFrame', frameData);
      
      // Send frame to server with auto_frame flag for annotation
      if (typeof sendJSON === 'function') {
        sendJSON({
          auto_frame: true,
          frame_data: frameData
        });
        console.log("Auto-captured frame sent for annotation");
      }
    } else {
      // We couldn't capture a frame on this interval
      console.warn("Failed to capture automatic frame");
      
      // Try to use any previously stored frame for annotation
      const lastFrame = sessionStorage.getItem('lastCapturedFrame');
      if (lastFrame && typeof sendJSON === 'function') {
        sendJSON({
          auto_frame: true,
          frame_data: lastFrame
        });
        console.log("Using previously captured frame for annotation");
      }
    }
  }, FRAME_CAPTURE_INTERVAL);
  
  console.log("Auto frame capture started");
}

function stopAutoFrameCapture() {
  if (frameCapture) {
    clearInterval(frameCapture);
    frameCapture = null;
    console.log("Auto frame capture stopped");
  }
}

// Function to capture the current video frame
function captureVideoFrame() {
  const videoElement = document.getElementById('surgery-video');
  
  if (!videoElement) {
    console.warn('Video element not found');
    return null;
  }
  
  if (!videoElement.src || videoElement.src === window.location.href) {
    console.warn('Video source is not set or invalid');
    return null;
  }
  
  // Check if the video element has a valid size
  const hasValidSize = videoElement.videoWidth > 0 && videoElement.videoHeight > 0;
  
  // You can capture frames from paused videos as well, as long as they've loaded
  // Only require the video to be playing if we don't have a current frame (initial load)
  const hasCurrentFrame = sessionStorage.getItem('lastCapturedFrame') !== null;
  
  if (videoElement.readyState < 2) { // HAVE_CURRENT_DATA (2) or higher needed
    console.warn('Video not ready for frame capture, readyState:', videoElement.readyState);
    return null;
  }
  
  // Allow capturing from paused videos if they've loaded a frame
  const canCapture = hasValidSize && videoElement.readyState >= 2;
  if (!canCapture) {
    console.warn('Video not ready for capture - either not loaded or no dimensions');
    
    // Return last captured frame if available
    const lastFrame = sessionStorage.getItem('lastCapturedFrame');
    if (lastFrame) {
      console.log("Using previous frame since video isn't ready for capture");
      return lastFrame;
    }
    
    // Create a placeholder frame if we can't get a real one
    return createPlaceholderFrame();
  }
  
  try {
    // Create a canvas element
    const canvas = document.createElement('canvas');
    
    // Check if video dimensions are available
    if (videoElement.videoWidth === 0 || videoElement.videoHeight === 0) {
      console.warn('Video dimensions are not available yet');
      // Return a placeholder frame instead of trying with default dimensions
      return createPlaceholderFrame();
    } else {
      canvas.width = videoElement.videoWidth;
      canvas.height = videoElement.videoHeight;
    }
    
    // Draw the current frame to the canvas
    const ctx = canvas.getContext('2d');
    
    // This try/catch specifically focuses on the drawing operation
    try {
      ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    } catch (drawErr) {
      console.error('Error drawing video frame to canvas:', drawErr);
      
      // Return the last frame we captured if available
      const lastFrame = sessionStorage.getItem('lastCapturedFrame');
      if (lastFrame) {
        console.log("Using last captured frame after drawing error");
        return lastFrame;
      }
      
      // Fallback to placeholder
      return createPlaceholderFrame();
    }
    
    // Convert to base64 data URL
    const dataURL = canvas.toDataURL('image/jpeg', 0.8);
    
    // Verify we got a valid data URL (should start with 'data:image/jpeg;base64,')
    if (!dataURL || !dataURL.startsWith('data:image/jpeg;base64,')) {
      console.error('Invalid data URL generated from canvas');
      
      // Try to use previous frame
      const lastFrame = sessionStorage.getItem('lastCapturedFrame');
      if (lastFrame) {
        console.log("Invalid data URL - using previous frame instead");
        return lastFrame;
      }
      
      return createPlaceholderFrame();
    }
    
    // Store successfully captured frame in session storage for fallback
    sessionStorage.setItem('lastCapturedFrame', dataURL);
    
    console.log('Frame captured successfully');
    return dataURL;
  } catch (err) {
    console.error('Error capturing video frame:', err);
    
    // Try to get the last stored frame
    const lastFrame = sessionStorage.getItem('lastCapturedFrame');
    if (lastFrame) {
      console.log("Using previously captured frame after error");
      return lastFrame;
    }
    
    // If nothing else works, create a placeholder
    return createPlaceholderFrame();
  }
}

// Generate summary function
function generateSummary() {
  const summaryContainer = document.getElementById('summary-container');
  
  // Show loading state
  summaryContainer.innerHTML = `
    <div class="flex justify-center items-center h-full">
      <div class="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-500"></div>
    </div>
  `;
  
  // Prevent multiple clicks
  const generateButton = document.getElementById('generate-summary-btn');
  if (generateButton) {
    generateButton.disabled = true;
    generateButton.classList.add('opacity-50');
  }
  
  // Add this function if it doesn't exist yet
  if (typeof displayFormattedPostOpNote !== 'function') {
    // Function to display formatted post-op note from JSON data
    window.displayFormattedPostOpNote = function(postOpNote, container) {
      if (!postOpNote) {
        container.innerHTML = `
          <div class="p-4 border border-red-700 rounded-lg">
            <h3 class="text-lg font-semibold mb-2 text-red-400">Error</h3>
            <p class="text-sm text-gray-300">No post-op note data was provided.</p>
          </div>
        `;
        return;
      }
    
      // Format procedure information
      const procInfo = postOpNote.procedure_information || {};
      
      // Create HTML content
      let html = `
        <div class="p-4 border border-dark-700 rounded-lg">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Procedure Information</h3>
          <div class="space-y-2">
            <p class="text-sm"><span class="font-medium text-gray-400">Type:</span> ${procInfo.procedure_type || 'Not specified'}</p>
            <p class="text-sm"><span class="font-medium text-gray-400">Date:</span> ${procInfo.date || 'Not specified'}</p>
            <p class="text-sm"><span class="font-medium text-gray-400">Duration:</span> ${procInfo.duration || 'Not specified'}</p>
            <p class="text-sm"><span class="font-medium text-gray-400">Surgeon:</span> ${procInfo.surgeon || 'Not specified'}</p>
          </div>
        </div>
      `;
      
      // Add findings section if available
      const findings = postOpNote.findings || [];
      if (findings.length > 0) {
        html += `
          <div class="p-4 border border-dark-700 rounded-lg mt-4">
            <h3 class="text-lg font-semibold mb-2 text-primary-400">Key Findings</h3>
            <ul class="list-disc list-inside space-y-1 text-sm">
        `;
        
        findings.forEach(finding => {
          if (finding && finding.trim()) {
            html += `<li>${finding}</li>`;
          }
        });
        
        html += `
            </ul>
          </div>
        `;
      }
      
      // Add timeline section if available
      const timeline = postOpNote.procedure_timeline || [];
      if (timeline.length > 0) {
        html += `
          <div class="p-4 border border-dark-700 rounded-lg mt-4">
            <h3 class="text-lg font-semibold mb-2 text-primary-400">Procedure Timeline</h3>
            <ul class="list-disc list-inside space-y-1 text-sm">
        `;
        
        timeline.forEach(event => {
          if (event.description && event.description.trim()) {
            html += `<li><span class='font-medium text-primary-300'>${event.time || 'Unknown'}</span>: ${event.description}</li>`;
          }
        });
        
        html += `
            </ul>
          </div>
        `;
      }
      
      // Add complications section if available
      const complications = postOpNote.complications || [];
      if (complications.length > 0) {
        html += `
          <div class="p-4 border border-dark-700 rounded-lg mt-4">
            <h3 class="text-lg font-semibold mb-2 text-primary-400">Complications</h3>
            <ul class="list-disc list-inside space-y-1 text-sm">
        `;
        
        complications.forEach(complication => {
          if (complication && complication.trim()) {
            html += `<li>${complication}</li>`;
          }
        });
        
        html += `
            </ul>
          </div>
        `;
      }
      
      // If no substantive content, show a message
      if (!findings.length && !timeline.length && !complications.length) {
        html += `
          <div class="p-4 border border-dark-700 rounded-lg mt-4">
            <h3 class="text-lg font-semibold mb-2 text-primary-400">Additional Information</h3>
            <p class="text-sm text-gray-300">
              Insufficient procedure data is available for a detailed summary. 
              For better results, add more annotations and notes during the procedure.
            </p>
          </div>
        `;
      }
      
      // Set the HTML
      container.innerHTML = html;
    };
  }
  
  // Gather data from notes
  const notesContainer = document.getElementById('notes-container');
  const notes = notesContainer ? Array.from(notesContainer.querySelectorAll('.bg-dark-800:not(.text-center)')) : [];
  
  // Gather data from annotations
  const annotationsContainer = document.getElementById('annotations-container');
  const annotations = annotationsContainer ? Array.from(annotationsContainer.querySelectorAll('.bg-dark-800:not(.text-center)')) : [];
  
  // Get current video duration
  const video = document.getElementById('surgery-video');
  const videoDuration = video ? formatTime(video.duration) : 'Unknown';
  
  // Extract note data to send to backend
  const noteData = notes.map(note => {
    const contentEl = note.querySelector('.note-content');
    const content = contentEl ? contentEl.textContent.trim() : '';
    const timeEl = note.querySelector('.text-xs.text-gray-500') || note.querySelector('[class*="text-"].text-gray-500');
    const time = timeEl ? timeEl.textContent.trim() : '';
    return { text: content, timestamp: time };
  }).filter(note => note.text && note.text !== 'Take a note');

  // Extract annotation data to send to backend
  const annotationData = annotations.map(ann => {
    // For annotations, the content is in a p tag with text-sm text-gray-300 classes
    const contentEl = ann.querySelector('p.text-sm.text-gray-300');
    const content = contentEl ? contentEl.textContent.trim() : '';
    
    // Get timestamp from the span
    const timeEl = ann.querySelector('.text-xs.text-gray-400');
    const time = timeEl ? timeEl.textContent.trim() : '';
    
    // Get surgical phase from the h3 tag
    const phaseEl = ann.querySelector('h3.text-lg.font-semibold');
    const phase = phaseEl ? phaseEl.textContent.trim() : '';
    
    // Get tools and anatomy from the badges
    const toolsEl = ann.querySelector('.badge.text-primary-300');
    const tools = toolsEl ? toolsEl.textContent.trim() : '';
    
    const anatomyEl = ann.querySelector('.badge.text-yellow-300');
    const anatomy = anatomyEl ? anatomyEl.textContent.trim() : '';
    
    return { 
      description: content, 
      timestamp: time,
      surgical_phase: phase,
      tools: tools ? [tools] : [],
      anatomy: anatomy ? [anatomy] : []
    };
  }).filter(ann => ann.description);

  // Log what we're sending to help with debugging
  console.log("Generating summary with:");
  console.log("- Notes:", noteData.length, noteData);
  console.log("- Annotations:", annotationData.length, annotationData);
  
  // Generate request to backend for summary generation with data
  fetch('/api/generate_post_op_note', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      request_type: 'generate_summary',
      notes: noteData,
      annotations: annotationData,
      video_duration: videoDuration
    })
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('Failed to generate summary');
    }
    return response.json();
  })
  .then(data => {
    console.log("Summary response:", data);
    
    if (data && data.post_op_note) {
      // Now we're receiving raw JSON instead of HTML
      displayFormattedPostOpNote(data.post_op_note, summaryContainer);
    } else if (data && data.summary) {
      // Backward compatibility with old API
      summaryContainer.innerHTML = `
        <div class="space-y-4">
          ${data.summary}
        </div>
      `;
    } else {
      // Fallback to client-side summary if the server doesn't return one
      fallbackGenerateSummary(notes, annotations, videoDuration, summaryContainer);
    }
    
    // Re-enable the button
    if (generateButton) {
      generateButton.disabled = false;
      generateButton.classList.remove('opacity-50');
    }
  })
  .catch(error => {
    console.error('Error generating summary:', error);
    
    // Fallback to client-side generation
    fallbackGenerateSummary(notes, annotations, videoDuration, summaryContainer);
    
    // Re-enable the button
    if (generateButton) {
      generateButton.disabled = false;
      generateButton.classList.remove('opacity-50');
    }
  });
}

// Fallback client-side summary generation
function fallbackGenerateSummary(notes, annotations, videoDuration, summaryContainer) {
  // Process after a small delay to show loading animation
  setTimeout(() => {
    // Extract procedure information
    const procedureType = determineTypeFromAnnotations(annotations);
    const phases = extractPhasesFromAnnotations(annotations);
    const keyEventsFromNotes = extractKeyEventsFromNotes(notes);
    const keyEventsFromAnnotations = extractKeyEventsFromAnnotations(annotations);
    
    // Combine key events and sort by timestamp
    const allKeyEvents = [...keyEventsFromNotes, ...keyEventsFromAnnotations]
      .filter(event => event && event.trim()) // Filter out empty events
      .sort((a, b) => {
        // Extract time values for comparison (assuming format like "00:12:18")
        const timeA = a.match(/(\d+:\d+)/);
        const timeB = b.match(/(\d+:\d+)/);
        if (!timeA || !timeB) return 0;
        
        const [minsA, secsA] = timeA[1].split(':').map(Number);
        const [minsB, secsB] = timeB[1].split(':').map(Number);
        
        return (minsA * 60 + secsA) - (minsB * 60 + secsB);
      });
    
    // Get actual notes content (that has meaningful data)
    const noteSummary = summarizeNotesByCategory(notes);
    const hasValidNotes = noteSummary && noteSummary.trim() && !noteSummary.includes('undefined');
    
    // Generate the HTML for the procedure info - always show this section
    let summaryHTML = `
      <div class="space-y-4">
        <div class="p-4 border border-dark-700 rounded-lg">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Procedure Information</h3>
          <div class="space-y-2">
            <p class="text-sm"><span class="font-medium text-gray-400">Type:</span> ${procedureType}</p>
            <p class="text-sm"><span class="font-medium text-gray-400">Duration:</span> ${videoDuration}</p>
            <p class="text-sm"><span class="font-medium text-gray-400">Phases:</span> ${phases.join(', ') || 'Not specified'}</p>
            <p class="text-sm"><span class="font-medium text-gray-400">Notes:</span> ${notes.length} | <span class="font-medium text-gray-400">Annotations:</span> ${annotations.length}</p>
          </div>
        </div>
    `;
    
    // Only add key events section if we have actual events
    if (allKeyEvents.length > 0) {
      summaryHTML += `
        <div class="p-4 border border-dark-700 rounded-lg">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Key Events</h3>
          <ul class="list-disc list-inside space-y-1 text-sm">
            ${allKeyEvents.map(event => `<li>${event}</li>`).join('')}
          </ul>
        </div>
      `;
    }
    
    // Only add notes summary if we have actual notes with content
    if (hasValidNotes) {
      summaryHTML += `
        <div class="p-4 border border-dark-700 rounded-lg">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Notes Summary</h3>
          <div class="space-y-3">
            ${noteSummary}
          </div>
        </div>
      `;
    }
    
    // Add a default message if we don't have much information to show
    if (!allKeyEvents.length && !hasValidNotes) {
      summaryHTML += `
        <div class="p-4 border border-dark-700 rounded-lg">
          <h3 class="text-lg font-semibold mb-2 text-primary-400">Additional Information</h3>
          <p class="text-sm text-gray-300">
            Not enough procedure data available for a detailed summary.
            Try adding more annotations and notes during the procedure for a more comprehensive summary.
          </p>
        </div>
      `;
    }
    
    // Close the main div
    summaryHTML += `</div>`;
    
    // Update the container with our generated HTML
    summaryContainer.innerHTML = summaryHTML;
  }, 1000);
}

// Helper function to determine procedure type from annotations
function determineTypeFromAnnotations(annotations) {
  // Default procedure type
  let procedureType = "Surgical Procedure";
  
  // Try to extract procedure type from annotations
  for (const annotation of annotations) {
    const content = annotation.textContent || '';
    if (content.toLowerCase().includes('laparoscopic')) {
      procedureType = "Laparoscopic Procedure";
      break;
    } else if (content.toLowerCase().includes('robotic')) {
      procedureType = "Robotic-Assisted Procedure";
      break;
    } else if (content.toLowerCase().includes('endoscopic')) {
      procedureType = "Endoscopic Procedure";
      break;
    } else if (content.toLowerCase().includes('open surgery')) {
      procedureType = "Open Surgery";
      break;
    }
  }
  
  return procedureType;
}

// Helper function to extract phases from annotations
function extractPhasesFromAnnotations(annotations) {
  const phases = new Set();
  
  for (const annotation of annotations) {
    const content = annotation.textContent || '';
    const phaseMatch = content.match(/Phase[:\s'"]+([^'"|\n]+)/i) || content.match(/phase[:\s'"]+([^'"|\n]+)/i);
    
    if (phaseMatch && phaseMatch[1]) {
      phases.add(phaseMatch[1].trim());
    }
  }
  
  return Array.from(phases);
}

// Helper function to extract key events from notes
function extractKeyEventsFromNotes(notes) {
  const keyEvents = [];
  
  for (const note of notes) {
    const title = note.querySelector('h3')?.textContent || '';
    const content = note.querySelector('.note-content')?.textContent || '';
    const timeElement = note.querySelector('.text-gray-400')?.textContent || '';
    const timeMatch = timeElement.match(/Video:\s*(\d+:\d+)/);
    const time = timeMatch ? timeMatch[1] : '';
    
    if (time && (title || content)) {
      let eventType = '';
      
      // Determine the type of event
      if (content.toLowerCase().includes('bleed') || title.toLowerCase().includes('bleed')) {
        eventType = 'Bleeding observed';
      } else if (content.toLowerCase().includes('tool') || title.toLowerCase().includes('tool') || 
                content.toLowerCase().includes('instrument') || title.toLowerCase().includes('instrument')) {
        eventType = 'Tool usage';
      } else if (content.toLowerCase().includes('incision') || title.toLowerCase().includes('incision') ||
                content.toLowerCase().includes('cut') || title.toLowerCase().includes('cut')) {
        eventType = 'Incision made';
      } else if (content.toLowerCase().includes('suture') || title.toLowerCase().includes('suture') ||
                content.toLowerCase().includes('stitch') || title.toLowerCase().includes('stitch')) {
        eventType = 'Suture placed';
      } else {
        eventType = 'Observation';
      }
      
      keyEvents.push(`${eventType} at ${time}`);
    }
  }
  
  return keyEvents;
}

// Helper function to extract key events from annotations
function extractKeyEventsFromAnnotations(annotations) {
  const keyEvents = [];
  
  for (const annotation of annotations) {
    const content = annotation.textContent || '';
    const timeMatch = content.match(/at time (\d+:\d+)/i);
    const time = timeMatch ? timeMatch[1] : '';
    
    // Look for key events in the annotation
    if (time || content.includes(':')) {
      let eventDesc = '';
      
      if (content.toLowerCase().includes('phase change')) {
        const phaseMatch = content.match(/Phase[:\s'"]+([^'"|\n]+)/i) || content.match(/phase[:\s'"]+([^'"|\n]+)/i);
        if (phaseMatch && phaseMatch[1]) {
          eventDesc = `Phase changed to '${phaseMatch[1].trim()}'`;
        } else {
          eventDesc = 'Phase changed';
        }
      } else if (content.toLowerCase().includes('tool')) {
        const toolMatch = content.match(/tool[s]?[:\s'"]+([^'"|\n]+)/i);
        if (toolMatch && toolMatch[1]) {
          eventDesc = `Tool used: ${toolMatch[1].trim()}`;
        } else {
          eventDesc = 'Tool usage noted';
        }
      } else if (content.toLowerCase().includes('anatomy')) {
        const anatomyMatch = content.match(/anatomy[:\s'"]+([^'"|\n]+)/i);
        if (anatomyMatch && anatomyMatch[1]) {
          eventDesc = `Anatomy identified: ${anatomyMatch[1].trim()}`;
        } else {
          eventDesc = 'Anatomy identified';
        }
      }
      
      if (eventDesc && time) {
        keyEvents.push(`${eventDesc} at ${time}`);
      }
    }
  }
  
  return keyEvents;
}

// Helper function to summarize notes by category
function summarizeNotesByCategory(notes) {
  const categories = {};
  
  // Group notes by category
  for (const note of notes) {
    // Get the category from the note
    const categoryElement = note.querySelector('.bg-primary-900\\/50') || 
                            note.querySelector('.text-xs.font-medium') ||
                            note.querySelector('[class*="text-"].font-medium');
    const category = categoryElement ? categoryElement.textContent.trim() : 'General';
    
    // Get the content - make sure it exists and isn't empty
    const contentElement = note.querySelector('.note-content');
    const content = contentElement?.textContent?.trim() || '';
    
    // Skip empty content
    if (!content) continue;
    
    // Initialize the category array if needed
    if (!categories[category]) {
      categories[category] = [];
    }
    
    // Add the content to the appropriate category
    categories[category].push(content);
  }
  
  // If we have no categories with content, return empty string
  if (Object.keys(categories).length === 0) {
    return '';
  }
  
  // Generate summary HTML
  let summary = '';
  for (const [category, contents] of Object.entries(categories)) {
    // Skip categories with no valid contents
    if (!contents.length) continue;
    
    // Filter out any undefined or empty contents
    const validContents = contents.filter(content => content && content.trim());
    
    // Skip if we have no valid contents after filtering
    if (!validContents.length) continue;
    
    summary += `
      <div class="mb-2">
        <h4 class="text-sm font-medium text-primary-300 mb-1">${category} (${validContents.length})</h4>
        <ul class="list-disc list-inside pl-2 text-xs text-gray-300 space-y-1">
          ${validContents.map(content => {
            // Truncate long content with ellipsis
            const displayContent = content.substring(0, 100) + (content.length > 100 ? '...' : '');
            return `<li>${displayContent}</li>`;
          }).join('')}
        </ul>
      </div>
    `;
  }
  
  return summary;
}

// Capture for note function (placeholder)
function captureForNote() {
  // In a real implementation, this would capture the current video frame
  
  // Show preview
  const previewContainer = document.getElementById('note-image-preview-container');
  const previewImage = document.getElementById('note-image-preview');
  
  // This would be replaced with the actual captured image
  previewImage.src = 'https://via.placeholder.com/640x360?text=Captured+Frame';
  previewContainer.classList.remove('hidden');
  
  showToast('Frame captured for note', 'success');
}

// Save manual note function
function saveManualNote() {
  const title = document.getElementById('note-title').value.trim();
  const content = document.getElementById('note-content').value.trim();
  const message = document.getElementById('note-message').value.trim();
  
  if (!title || !content) {
    showToast('Please enter a title and content for your note', 'error');
    return;
  }
  
  // Format the note for processing
  const noteText = `Note: ${title}. ${content}`;
  
  // Use our enhanced addNote function to create the note
  addNote(noteText, `Take a note about ${title}`);
  
  // Close modal
  const modal = document.getElementById('addNoteModal');
  closeModal(modal);
  
  // Reset the form
  resetNoteForm();
  
  showToast('Note saved successfully', 'success');
  
  // If there's a message, send it to the chat
  if (message) {
    addMessageToChat(message, 'user');
    // Send the message to the backend if needed
    sendMessageToBackend(message);
  }
}

// Function to fetch and display videos
function loadVideos() {
  const videoList = document.getElementById('video-list');
  const videoLoading = document.getElementById('video-loading');
  const noVideos = document.getElementById('no-videos');
  
  // Show loading state
  if (videoLoading) {
    videoLoading.style.display = 'block';
  }
  if (noVideos) {
    noVideos.style.display = 'none';
  }
  
  // Clear existing videos
  const existingVideos = videoList.querySelectorAll('.video-item');
  existingVideos.forEach(item => item.remove());
  
  // Fetch videos from the server
  fetch('/api/videos')
    .then(response => {
      if (!response.ok) {
        throw new Error('Failed to fetch videos');
      }
      return response.json();
    })
    .then(data => {
      if (videoLoading) {
        videoLoading.style.display = 'none';
      }
      
      if (data.videos && data.videos.length > 0) {
        // Display videos
        data.videos.forEach((video, index) => {
          const videoItem = document.createElement('div');
          videoItem.className = 'video-item p-0 bg-dark-800 rounded-xl border border-dark-600 hover:border-primary-500 transition-all duration-200 cursor-pointer overflow-hidden group';
          
          // Format file size
          const sizeInMB = (video.size / (1024 * 1024)).toFixed(2);
          
          // Format date
          const date = new Date(video.modified * 1000);
          const formattedDate = date.toLocaleDateString(undefined, {
            month: 'short',
            day: 'numeric',
            year: 'numeric'
          });
          
          // Get a random duration for preview purposes (in real app this would come from the video metadata)
          const duration = `${Math.floor(Math.random() * 10) + 1}:${Math.floor(Math.random() * 60).toString().padStart(2, '0')}`;
          
          // Create a unique ID for this video item
          const videoItemId = `video-item-${index}`;
          
          videoItem.innerHTML = `
            <div class="flex flex-col sm:flex-row sm:items-center">
              <div class="bg-dark-900 h-full min-h-24 sm:w-40 flex items-center justify-center p-3 relative">
                <div class="absolute inset-0 bg-gradient-to-r from-primary-900/30 to-primary-700/30 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                <i class="fas fa-film text-primary-400 text-4xl relative z-10"></i>
                <span class="absolute bottom-2 right-2 text-xs text-white bg-dark-900/80 px-2 py-1 rounded">${duration}</span>
              </div>
              <div class="flex-grow p-4">
                <div class="flex items-start justify-between">
                  <h4 class="text-lg font-medium text-white group-hover:text-primary-300 transition-colors duration-200 truncate max-w-[80%]">${video.filename}</h4>
                  <div class="badge bg-dark-700 text-xs text-gray-300 ml-2">${sizeInMB} MB</div>
                </div>
                <div class="flex items-center mt-2 text-sm text-gray-400">
                  <i class="fas fa-calendar-alt mr-1 text-primary-500"></i>
                  <span>${formattedDate}</span>
                </div>
              </div>
              <div class="flex items-center justify-end p-3 pr-4 bg-dark-800 sm:bg-transparent">
                <button class="btn px-4 py-2 rounded-lg bg-gradient-to-r from-primary-600 to-primary-700 hover:from-primary-500 hover:to-primary-600 text-white transform hover:scale-105 transition-all duration-200 shadow-md select-video-btn" data-filename="${video.filename}" data-item-id="${videoItemId}">
                  <i class="fas fa-play mr-2"></i> Play
                </button>
              </div>
            </div>
          `;
          
          videoItem.id = videoItemId;
          videoList.appendChild(videoItem);
          
          // Add click event to select button
          const selectBtn = videoItem.querySelector('.select-video-btn');
          if (selectBtn) {
            selectBtn.addEventListener('click', (e) => {
              e.preventDefault();
              // Add loading indicator to button
              selectBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Loading...';
              selectBtn.disabled = true;
              // Call select function
              selectVideo(video.filename);
            });
          }
          
          // Add click event to entire item (except the button)
          videoItem.addEventListener('click', (e) => {
            // If not clicking the button itself
            if (!e.target.closest('.select-video-btn')) {
              // Find and click the select button
              const btn = videoItem.querySelector('.select-video-btn');
              if (btn && !btn.disabled) {
                btn.click();
              }
            }
          });
        });
      } else {
        // Show no videos message
        if (noVideos) {
          noVideos.style.display = 'block';
        }
      }
    })
    .catch(error => {
      console.error('Error fetching videos:', error);
      
      if (videoLoading) {
        videoLoading.style.display = 'none';
      }
      
      if (noVideos) {
        noVideos.style.display = 'block';
        noVideos.innerHTML = `
          <div class="bg-dark-800 rounded-xl p-6 border border-red-900">
            <i class="fas fa-exclamation-circle text-5xl mb-4 text-red-500"></i>
            <p class="text-lg text-red-300">Error loading videos</p>
            <p class="text-sm text-gray-400 mt-2">${error.message}</p>
          </div>
        `;
      }
    });
}

// Function to select a video
function selectVideo(filename) {
  // Close the modal
  const modal = document.getElementById('videoSelectModal');
  closeModal(modal);
  
  // Show loading state
  showToast('Loading video...', 'info');
  
  // Send request to server
  fetch('/api/select_video', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ filename })
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('Failed to select video');
    }
    return response.json();
  })
  .then(data => {
    // Update video source
    const videoElement = document.getElementById('surgery-video');
    if (videoElement && data.video_src) {
      // Stop any current playback
      try {
        videoElement.pause();
      } catch (e) {
        console.warn("Could not pause video:", e);
      }
      
      // Set new source and load (but don't play automatically)
      videoElement.src = data.video_src;
      videoElement.load();
      
      // Enable autoplay
      videoElement.autoplay = true;
      
      // Show success message
      showToast('Video loaded successfully!', 'success');
      
      // Enable the microphone button
      enableMicButton();
      
      // Reset the current phase 
      const phaseElement = document.getElementById('current-phase');
      if (phaseElement) {
        phaseElement.textContent = 'Undefined';
      }
      
      // Clear any existing annotations
      const annotationsContainer = document.getElementById('annotations-container');
      if (annotationsContainer) {
        annotationsContainer.innerHTML = `
          <div class="text-center text-gray-400 p-5">
            <i class="fas fa-tag fa-3x mb-3"></i>
            <p>No annotations available yet. Annotations will appear here as they are generated.</p>
          </div>
        `;
      }
      
      // Update the count
      const annotationCount = document.querySelector('.annotation-count');
      if (annotationCount) {
        annotationCount.textContent = '0';
      }
    }
  })
  .catch(error => {
    console.error('Error selecting video:', error);
    showToast('Failed to load video: ' + error.message, 'error');
  });
}

function closeModal(modal) {
  if (!modal) return;
  
  modal.classList.remove('show');
  modal.classList.add('closing');
  
  setTimeout(() => {
    modal.style.display = 'none'; // Reset display property
    modal.classList.remove('closing');
    document.body.classList.remove('overflow-hidden');
  }, 300);
}

// Video upload functionality
function uploadVideo() {
  const fileInput = document.getElementById('video-upload');
  const file = fileInput.files[0];
  const uploadForm = document.getElementById('video-upload-form');
  const uploadButton = uploadForm?.querySelector('button[type="button"]');
  
  if (!file) {
    showToast('Please select a video file first', 'error');
    return;
  }
  
  // Create FormData object
  const formData = new FormData();
  formData.append('video', file);
  
  // Show loading state
  if (uploadButton) {
    uploadButton.disabled = true;
    uploadButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Uploading...';
  }
  showToast('Uploading video...', 'info');
  
  // Use fetch to upload the file
  fetch('/api/upload_video', {
    method: 'POST',
    body: formData
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('Network response was not ok');
    }
    return response.json();
  })
  .then(data => {
    // Handle successful upload with the actual filename
    const displayName = data.filename || 'video';
    showToast(`"${displayName}" uploaded successfully!`, 'success');
    
    // Reset file name display
    const fileNameElement = fileInput.closest('.relative')?.querySelector('.file-name');
    if (fileNameElement) {
      fileNameElement.textContent = 'Select video file...';
    }
    
    // Reset upload button
    if (uploadButton) {
      uploadButton.disabled = false;
      uploadButton.innerHTML = '<i class="fas fa-upload mr-2"></i> Upload';
    }
    
    // Clear the file input
    fileInput.value = '';
    
    // Update the button to show "Uploaded!" temporarily
    if (uploadButton) {
      uploadButton.classList.add('bg-green-600', 'border-green-700');
      uploadButton.innerHTML = '<i class="fas fa-check mr-2"></i> Uploaded!';
      setTimeout(() => {
        uploadButton.classList.remove('bg-green-600', 'border-green-700');
        uploadButton.innerHTML = '<i class="fas fa-upload mr-2"></i> Upload';
      }, 2000);
    }
    
    // Load the video but don't autoplay
    if (data && data.video_src) {
      const videoElement = document.getElementById('surgery-video');
      if (videoElement) {
        // Pause any current video first
        try {
          videoElement.pause();
        } catch (e) {
          console.warn("Could not pause video:", e);
        }
        
        // Set new source with autoplay
        videoElement.src = data.video_src;
        videoElement.load();
        videoElement.autoplay = true;
        
        // Enable the microphone button
        enableMicButton();
        
        // Reset the current phase display
        const phaseElement = document.getElementById('current-phase');
        if (phaseElement) {
          phaseElement.textContent = 'Undefined';
        }
        
        // Clear existing annotations
        const annotationsContainer = document.getElementById('annotations-container');
        if (annotationsContainer) {
          annotationsContainer.innerHTML = `
            <div class="text-center text-gray-400 p-5">
              <i class="fas fa-tag fa-3x mb-3"></i>
              <p>No annotations available yet. Annotations will appear here as they are generated.</p>
            </div>
          `;
        }
        
        // Reset annotation count
        const annotationCount = document.querySelector('.annotation-count');
        if (annotationCount) {
          annotationCount.textContent = '0';
        }
        
        // Clear existing notes
        const notesContainer = document.getElementById('notes-container');
        if (notesContainer) {
          notesContainer.innerHTML = `
            <div class="text-center text-gray-400 p-5">
              <i class="fas fa-sticky-note fa-3x mb-3"></i>
              <p>No notes available yet. You can add notes manually or ask the assistant to take notes for you.</p>
            </div>
          `;
        }
        
        // Reset notes count
        const notesCount = document.querySelector('.notes-count');
        if (notesCount) {
          notesCount.textContent = '0';
        }
      }
    }
  })
  .catch(error => {
    console.error('Error uploading video:', error);
    showToast('Failed to upload video. Please try again.', 'error');
    
    // Reset upload button
    if (uploadButton) {
      uploadButton.disabled = false;
      uploadButton.innerHTML = '<i class="fas fa-upload mr-2"></i> Upload';
    }
  });
}