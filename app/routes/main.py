import os
import time
import traceback
import threading
import logging
import re
import json
from flask import (
    Blueprint, flash, redirect, render_template, request, 
    url_for, current_app, send_from_directory, jsonify, session, copy_current_request_context
)
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app.models.material_takeoff import MaterialTakeoffAnalyzer
from flask import current_app as app

bp = Blueprint('main', __name__)

# Global dictionary to track analysis status
analysis_tasks = {}
# Thread dictionary to keep track of running threads
threads = {}
# Dictionary to track update threads
update_threads = {}

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@bp.route('/favicon.ico')
def favicon():
    """Serve the favicon directly."""
    return send_from_directory(
        os.path.join(current_app.root_path, 'static'),
        'favicon.ico', 
        mimetype='image/vnd.microsoft.icon'
    )

@bp.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@bp.route('/dashboard')
@login_required
def dashboard():
    """Render the dashboard page."""
    return render_template('dashboard.html')

@bp.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle file upload."""
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Secure the filename and add timestamp to avoid overwrites
            timestamp = int(time.time())
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            
            # Ensure the upload directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save the file
            file.save(file_path)
            
            current_app.logger.info(f"File uploaded: {filename}")
            
            # Initialize analysis task status
            analysis_tasks[filename] = {
                'status': 'pending',
                'error': None,
                'results': None
            }
            
            # Redirect to loading page
            return redirect(url_for('main.loading', filename=filename))
        
        else:
            flash('File type not allowed. Please upload an IFC file.')
            return redirect(url_for('main.index'))
    
    except Exception as e:
        current_app.logger.error(f"Error uploading file: {str(e)}\n{traceback.format_exc()}")
        flash(f'An unexpected error occurred during upload: {str(e)}')
        return redirect(url_for('main.index'))

@bp.route('/loading/<filename>')
@login_required
def loading(filename):
    """Show loading page and start analysis in background."""
    try:
        # Sanitize filename to prevent path traversal
        filename = os.path.basename(filename)
        
        # Check if file exists
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            flash('File not found')
            current_app.logger.warning(f"File not found: {filename}")
            return redirect(url_for('main.index'))
        
        # Save the current application context for the background thread
        upload_folder = current_app.config['UPLOAD_FOLDER']
        app_instance = current_app._get_current_object()
        
        # Start analysis in a background thread if not already started
        if filename not in analysis_tasks:
            # Initialize a new task if it doesn't exist
            analysis_tasks[filename] = {
                'status': 'pending',
                'error': None,
                'phase': 'initializing',
                'phase_description': 'Preparing to analyze file',
                'start_time': time.time()
            }
            current_app.logger.info(f"Created new task for {filename} in loading route")
            
        # Start the analysis if it's pending
        if analysis_tasks.get(filename, {}).get('status') == 'pending':
            analysis_tasks[filename]['status'] = 'running'
            
            # Clean up any old thread with the same name
            if filename in threads and threads[filename].is_alive():
                current_app.logger.warning(f"Thread for {filename} already exists. Status update only.")
            else:
                # Create and start new thread with the application context
                thread = threading.Thread(
                    target=analyze_file_task, 
                    args=(filename, upload_folder, app_instance),
                    name=f"analysis_{filename}"
                )
                thread.daemon = True
                thread.start()
                threads[filename] = thread
                current_app.logger.info(f"Started analysis thread for {filename}")
        
        # Return loading template
        return render_template('loading.html', filename=filename)
    
    except Exception as e:
        current_app.logger.error(f"Error in loading route: {str(e)}\n{traceback.format_exc()}")
        flash(f'An unexpected error occurred: {str(e)}')
        return redirect(url_for('main.index'))

def analyze_file_task(filename, upload_folder, app):
    """Background task to analyze the file."""
    # Create a Flask application context for this thread
    with app.app_context():
        # Set up thread-specific logging safely
        thread_logger = logging.getLogger(f"analysis_thread_{filename}")
        thread_logger.setLevel(logging.INFO)
        
        # Add a console handler to make sure logs appear somewhere if file handler fails
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        thread_logger.addHandler(console_handler)
        
        try:
            # Sanitize filename to prevent path traversal
            filename = os.path.basename(filename)
            
            file_path = os.path.join(upload_folder, filename)
            thread_logger.info(f"Processing file at: {file_path}")
            thread_logger.info(f"Upload folder: {upload_folder}")
            
            # Get the results folder from config
            results_folder = app.config.get('RESULTS_FOLDER', upload_folder)
            thread_logger.info(f"Results will be saved to: {results_folder}")
            
            # Check for PythonAnywhere environment
            is_pythonanywhere = app.config.get('IS_PYTHONANYWHERE', False)
            if is_pythonanywhere:
                thread_logger.info("Running on PythonAnywhere")
            
            # Verify directories exist and are writable
            os.makedirs(upload_folder, exist_ok=True)
            os.makedirs(results_folder, exist_ok=True)
            
            # Test write permissions for upload folder
            try:
                test_file = os.path.join(upload_folder, '.test_write')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                thread_logger.info("Upload folder is writable")
            except Exception as e:
                thread_logger.error(f"Upload folder is not writable: {str(e)}")
                thread_logger.error(traceback.format_exc())
            
            # Test write permissions for results folder
            try:
                test_file = os.path.join(results_folder, '.test_write')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                thread_logger.info("Results folder is writable")
            except Exception as e:
                thread_logger.error(f"Results folder is not writable: {str(e)}")
                thread_logger.error(traceback.format_exc())
                # Try temp directory as fallback
                import tempfile
                results_folder = tempfile.gettempdir()
                thread_logger.info(f"Using temp directory instead: {results_folder}")
            
            # Check if file exists
            if not os.path.exists(file_path):
                error_msg = f"IFC file not found at path: {file_path}"
                thread_logger.error(error_msg)
                analysis_tasks[filename] = {
                    'status': 'failed',
                    'error': error_msg,
                    'phase': 'error',
                    'phase_description': 'File not found'
                }
                return
            
            # Initial update to status - set the analysis start time
            analysis_start_time = time.time()
            analysis_tasks[filename] = {
                **analysis_tasks[filename],
                'status': 'running',
                'processed_elements': 0,
                'detailed_processing_elements': 0,  # Initialize detailed processing counter
                'phase': 'initializing',
                'phase_description': 'Loading IFC file',
                'start_time': analysis_start_time
            }
            thread_logger.info(f"Starting analysis for {filename}")
            
            # Create analyzer instance
            analyzer = MaterialTakeoffAnalyzer(file_path)
            
            # Set up log message interceptor to track detailed processing progress
            class DetailedProcessingLogHandler(logging.Handler):
                def emit(self, record):
                    message = record.getMessage()
                    # Look for detailed processing messages of the format: "Processed X/Y elements (Z%)"
                    match = re.search(r"Processed (\d+)/(\d+) elements", message)
                    if match:
                        processed = int(match.group(1))
                        total = int(match.group(2))
                        # Update the task with detailed processing progress
                        if filename in analysis_tasks:
                            analysis_tasks[filename]['detailed_processing_elements'] = processed
            
            # Add the detailed processing tracker log handler
            detailed_log_handler = DetailedProcessingLogHandler()
            analyzer.logger.addHandler(detailed_log_handler)
            
            # Update status to reflect file is loaded
            analysis_tasks[filename]['phase'] = 'analyzing'
            analysis_tasks[filename]['phase_description'] = 'Preparing to analyze elements'
            thread_logger.info(f"Created analyzer for {filename}")
            
            # Get total element count
            total_elements = len(analyzer.ifc_file.by_type('IfcProduct'))
            
            # Update task with element count and phase
            analysis_tasks[filename].update({
                'total_elements': total_elements,
                'phase_description': f'Analyzing {total_elements} elements'
            })
            
            thread_logger.info(f"IFC file contains {total_elements} elements")
            
            # Store the original analyze_all_elements method
            original_analyze_all_elements = analyzer.analyze_all_elements
            
            # Create a proxy method to simulate slower progress updates
            def analyze_all_elements_with_progress():
                # Reset processed elements counter
                analysis_tasks[filename]['processed_elements'] = 0
                analysis_tasks[filename]['total_elements'] = total_elements
                
                # Start element processing timer
                elements_start_time = time.time()
                thread_logger.info(f"Starting detailed analysis of {total_elements} elements")
                
                # Instead of immediately counting all elements, we'll do a separate, faster pass
                # to get accurate element count, and then simulate slower progress during actual analysis
                elements_list = list(analyzer.ifc_file.by_type('IfcProduct'))
                
                # Launch a background thread to update progress while analysis runs
                def update_progress_during_analysis():
                    # Estimate a reasonable analysis time based on element count
                    # Usually analysis takes about 5-10 seconds per 5000 elements
                    estimated_analysis_seconds = max(8, total_elements / 1000)
                    
                    # Calculate how many progress updates to do (1 per 2% progress)
                    update_count = 50  # Update approximately 50 times (2% increments)
                    sleep_time = estimated_analysis_seconds / update_count
                    
                    # Sleep to simulate initialization
                    time.sleep(1)
                    
                    # Update progress at regular intervals
                    for i in range(1, update_count+1):
                        try:
                            # Calculate simulated progress
                            simulated_progress = int((i / update_count) * total_elements)
                            
                            # Update progress in task status
                            analysis_tasks[filename]['processed_elements'] = min(simulated_progress, total_elements)
                            
                            # Log progress
                            if i % 5 == 0:  # Log every 10% progress
                                thread_logger.info(
                                    f"Analysis progress: {simulated_progress}/{total_elements} elements "
                                    f"({(simulated_progress/total_elements)*100:.1f}%)"
                                )
                            
                            # Sleep until next update
                            time.sleep(sleep_time)
                            
                            # Stop if analysis is complete
                            if analysis_tasks[filename].get('analysis_complete'):
                                break
                                
                        except Exception as e:
                            thread_logger.warning(f"Error updating progress: {str(e)}")
                
                # Start progress updater thread
                progress_thread = threading.Thread(
                    target=update_progress_during_analysis,
                    name=f"progress_{filename}"
                )
                progress_thread.daemon = True
                progress_thread.start()
                
                # Call the original method to do the actual analysis
                try:
                    result = original_analyze_all_elements()
                finally:
                    # Mark analysis as complete to stop progress thread
                    analysis_tasks[filename]['analysis_complete'] = True
                    analysis_tasks[filename]['processed_elements'] = total_elements
                
                # Wait for progress thread to finish
                if progress_thread.is_alive():
                    progress_thread.join(timeout=2)
                
                return result
            
            # Replace the method with our progress-tracking version
            analyzer.analyze_all_elements = analyze_all_elements_with_progress
            
            # Analyze all elements
            thread_logger.info(f"Analyzing elements for {filename}")
            analyze_start_time = time.time()
            analyzer.analyze_all_elements()
            analyze_end_time = time.time()
            analyze_duration = analyze_end_time - analyze_start_time
            
            # Calculate and log analysis time
            thread_logger.info(
                f"Finished analyzing {total_elements} elements in {analyze_duration:.1f} seconds "
                f"(avg: {total_elements/analyze_duration:.1f} elements/second)"
            )
            
            # Ensure processed_elements is updated to final count
            analysis_tasks[filename]['processed_elements'] = total_elements
            
            # Update status to next phase
            analysis_tasks[filename].update({
                'phase': 'generating_results',
                'phase_description': 'Generating summary and reports',
                'element_processing_complete': True,
                'element_processing_time': analyze_duration
            })
            
            # Generate summary
            base_name = os.path.splitext(filename)[0]
            thread_logger.info(f"Base name for results: {base_name}")
            thread_logger.info(f"Saving results to: {results_folder}")
            
            # Save results to the results folder using the updated method
            try:
                # Save all results formats, passing results_folder explicitly
                thread_logger.info(f"Saving analysis results for {filename}")
                result_files = analyzer.save_results(output_format='all', results_folder=results_folder)
                
                # Verify returned file info contains expected files
                if not result_files or 'json_file' not in result_files:
                    thread_logger.error(f"No result files returned after save operation: {result_files}")
                    analysis_tasks[filename].update({
                        'status': 'failed',
                        'error': "Failed to save analysis results - no files were returned"
                    })
                    return
                
                # Get output directory from result
                output_dir = result_files.get('output_dir', results_folder)
                thread_logger.info(f"Results were saved to: {output_dir}")
                
                # List all files in directory to confirm what was saved
                try:
                    dir_files = os.listdir(output_dir)
                    thread_logger.info(f"Files in results directory: {dir_files}")
                except Exception as e:
                    thread_logger.error(f"Error listing files in output directory: {str(e)}")
                
                # Update task status with result file locations
                analysis_tasks[filename].update({
                    'status': 'completed',
                    'phase': 'complete',
                    'phase_description': 'Analysis complete',
                    'total_analysis_time': time.time() - analysis_start_time,
                    'results': {
                        **result_files,
                        'output_dir': output_dir  # Include the final output directory used
                    }
                })
                
                # Final verification that the result files exist
                json_file = result_files.get('json_file', '')
                if json_file:
                    json_path = os.path.join(output_dir, json_file)
                    if os.path.exists(json_path):
                        thread_logger.info(f"Verified JSON result file at: {json_path}")
                    else:
                        thread_logger.error(f"JSON result file not found at: {json_path}")
                
                thread_logger.info(f"Analysis and result generation complete for {filename}")
                
            except Exception as e:
                thread_logger.error(f"Error saving result files: {str(e)}\n{traceback.format_exc()}")
                analysis_tasks[filename].update({
                    'status': 'failed',
                    'error': f"Failed to save analysis results: {str(e)}",
                    'phase': 'error',
                    'phase_description': 'Error generating result files'
                })
                
        except Exception as e:
            error_message = f"Analysis failed: {str(e)}"
            thread_logger.error(f"{error_message}\n{traceback.format_exc()}")
            
            # Update task status
            analysis_tasks[filename].update({
                'status': 'failed',
                'error': error_message,
                'phase': 'error',
                'phase_description': 'Analysis failed due to an error'
            })
        
        finally:
            # Cleanup thread reference
            if filename in threads:
                del threads[filename]

@bp.route('/analyze/<filename>')
@login_required
def analyze(filename):
    """View the analysis results for a file."""
    try:
        # Sanitize filename to prevent path traversal
        filename = os.path.basename(filename)
        
        # Debug info - log all important paths
        current_app.logger.info(f"Analyze route called for file: {filename}")
        current_app.logger.info(f"Upload folder config: {current_app.config['UPLOAD_FOLDER']}")
        current_app.logger.info(f"Results folder config: {current_app.config.get('RESULTS_FOLDER')}")
        current_app.logger.info(f"PythonAnywhere: {current_app.config.get('IS_PYTHONANYWHERE', False)}")
        
        # Check if the analysis task exists
        if filename not in analysis_tasks:
            current_app.logger.warning(f"Analysis task not found for {filename}. Available tasks: {list(analysis_tasks.keys())}")
            flash('Analysis task not found. Please try uploading and analyzing the file again.')
            return redirect(url_for('main.index'))
        
        # Get the task status
        task = analysis_tasks[filename]
        status = task.get('status', 'unknown')
        current_app.logger.info(f"Analysis status for {filename}: {status}")
        
        if status == 'failed':
            error = task.get('error', 'Unknown error')
            current_app.logger.error(f"Analysis failed for {filename}: {error}")
            flash(f'Analysis failed: {error}')
            return redirect(url_for('main.index'))
        
        if status == 'running':
            # Redirect to loading page if still running
            current_app.logger.info(f"Analysis still running for {filename}, redirecting to loading page")
            return redirect(url_for('main.loading', filename=filename))
        
        if status == 'completed':
            results = task.get('results', {})
            
            # Log the results for debugging
            current_app.logger.info(f"Results for {filename}: {results}")
            
            # Get folders from config
            upload_folder = current_app.config['UPLOAD_FOLDER']
            results_folder = current_app.config.get('RESULTS_FOLDER', upload_folder)
            output_dir = results.get('output_dir', results_folder)  # Use the directory from results if available
            
            current_app.logger.info(f"Upload folder: {upload_folder}")
            current_app.logger.info(f"Results folder: {results_folder}")
            current_app.logger.info(f"Output directory used: {output_dir}")
            
            # Prioritize JSON file first
            json_file = results.get('json_file', '')
            json_path = os.path.join(output_dir, json_file)  # Try the output directory first
            current_app.logger.info(f"Checking for JSON file at: {json_path}")
            
            # List of potential directories to check
            directories_to_check = [
                output_dir,
                results_folder,
                upload_folder
            ]
            
            # Check if JSON file exists in any of the directories
            json_path_found = None
            for directory in directories_to_check:
                potential_path = os.path.join(directory, json_file)
                current_app.logger.info(f"Checking in directory: {directory}")
                
                if os.path.exists(potential_path):
                    json_path_found = potential_path
                    current_app.logger.info(f"Found JSON file at: {json_path_found}")
                    break
            
            # If not found directly, try to find any matching file
            if not json_path_found:
                current_app.logger.error(f"JSON file not found at expected locations")
                
                # Try to find any file with a matching pattern in each directory
                base_name = os.path.splitext(filename)[0]
                
                for directory in directories_to_check:
                    try:
                        # List all files in directory
                        dir_files = os.listdir(directory)
                        current_app.logger.info(f"Files in {directory}: {dir_files}")
                        
                        # Try to find any matching JSON file
                        potential_json_files = [f for f in dir_files if f.startswith(base_name) and f.endswith('.json')]
                        if potential_json_files:
                            current_app.logger.info(f"Found potential JSON files: {potential_json_files}")
                            # Use the first matching file
                            json_file = potential_json_files[0]
                            json_path_found = os.path.join(directory, json_file)
                            current_app.logger.info(f"Using alternative JSON file: {json_path_found}")
                            
                            # Update the results dictionary
                            results['json_file'] = json_file
                            results['output_dir'] = directory
                            task['results'] = results
                            break
                    except Exception as e:
                        current_app.logger.error(f"Error checking directory {directory}: {str(e)}")
                
                # If still not found, return error
                if not json_path_found:
                    flash('Results file is missing. Please try analyzing the file again.')
                    return redirect(url_for('main.index'))
            
            # Use the found JSON path
            json_path = json_path_found
            output_dir = os.path.dirname(json_path)
            
            # Get additional result files if they exist
            excel_file = results.get('excel_file', '')
            summary_file = results.get('summary_file', '')
            details_file = results.get('details_file', '')
            
            # Check for files in the same directory as the JSON file
            excel_path = os.path.join(output_dir, excel_file) if excel_file else None
            summary_path = os.path.join(output_dir, summary_file) if summary_file else None
            details_path = os.path.join(output_dir, details_file) if details_file else None
            
            excel_exists = excel_path and os.path.exists(excel_path)
            summary_exists = summary_path and os.path.exists(summary_path)
            details_exists = details_path and os.path.exists(details_path)
            
            current_app.logger.info(f"Excel file exists: {excel_exists}, path: {excel_path}")
            current_app.logger.info(f"Summary file exists: {summary_exists}, path: {summary_path}")
            current_app.logger.info(f"Details file exists: {details_exists}, path: {details_path}")
            
            # Read JSON data to display on results page
            try:
                with open(json_path, 'r') as f:
                    json_data = json.load(f)
                    current_app.logger.info(f"Successfully loaded JSON data from {json_path}")
            except Exception as e:
                current_app.logger.error(f"Error reading JSON file: {str(e)}\n{traceback.format_exc()}")
                flash(f'Error reading results file: {str(e)}')
                return redirect(url_for('main.index'))
            
            # Save the paths relative to output_dir for download
            json_rel_path = os.path.basename(json_path) if json_path else None
            excel_rel_path = os.path.basename(excel_path) if excel_path else None
            summary_rel_path = os.path.basename(summary_path) if summary_path else None
            details_rel_path = os.path.basename(details_path) if details_path else None
            
            return render_template(
                'results.html', 
                filename=filename,
                json_file=json_rel_path,
                excel_file=excel_rel_path,
                summary_file=summary_rel_path,
                details_file=details_rel_path,
                json_data=json_data,
                has_warning=task.get('warning', None)
            )
        
        # Default case - should not reach here
        flash('Unknown analysis status')
        return redirect(url_for('main.index'))
    
    except Exception as e:
        current_app.logger.error(f"Error in analyze route: {str(e)}\n{traceback.format_exc()}")
        flash(f'An unexpected error occurred: {str(e)}')
        return redirect(url_for('main.index'))

@bp.route('/download/<filename>')
@login_required
def download_file(filename):
    """Download a file."""
    try:
        # Sanitize filename to prevent path traversal
        filename = os.path.basename(filename)
        
        # Get folders from config
        upload_folder = current_app.config['UPLOAD_FOLDER']
        results_folder = current_app.config.get('RESULTS_FOLDER', upload_folder)
        
        # Create a list of directories to check for the file
        directories = [results_folder, upload_folder]
        
        # Check if file exists in any of the directories
        found_path = None
        for directory in directories:
            file_path = os.path.join(directory, filename)
            if os.path.exists(file_path):
                found_path = file_path
                found_dir = directory
                current_app.logger.info(f"Found requested file {filename} in {directory}")
                break
        
        # If file not found in any directory
        if not found_path:
            current_app.logger.error(f"Download file not found: {filename}")
            current_app.logger.info(f"Checked directories: {directories}")
            
            # Try to list directory contents for debugging
            for directory in directories:
                try:
                    dir_files = os.listdir(directory)
                    current_app.logger.info(f"Files in {directory}: {dir_files}")
                except Exception as e:
                    current_app.logger.error(f"Error listing files in {directory}: {str(e)}")
            
            flash(f'File not found: {filename}')
            return redirect(url_for('main.index'))
        
        # Return the file from the directory where it was found
        return send_from_directory(
            found_dir,
            filename,
            as_attachment=True
        )
    except FileNotFoundError:
        current_app.logger.error(f"Download file not found: {filename}")
        flash(f'File not found: {filename}')
        return redirect(url_for('main.index'))
    except Exception as e:
        current_app.logger.error(f"Error downloading file: {filename} - {str(e)}\n{traceback.format_exc()}")
        flash(f'Error downloading file: {str(e)}')
        return redirect(url_for('main.index'))

@bp.route('/test_paths')
def test_paths():
    """Test route to verify directory paths."""
    # Get configured paths
    upload_folder = current_app.config['UPLOAD_FOLDER']
    results_folder = current_app.config.get('RESULTS_FOLDER', upload_folder)
    is_pythonanywhere = current_app.config.get('IS_PYTHONANYWHERE', False)
    
    # Log paths for verification
    current_app.logger.info(f"Upload folder: {upload_folder}")
    current_app.logger.info(f"Results folder: {results_folder}")
    current_app.logger.info(f"IS_PYTHONANYWHERE: {is_pythonanywhere}")
    
    # Create directories if they don't exist
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)
    
    # Test write access to directories
    upload_test_file = os.path.join(upload_folder, '.test_write')
    results_test_file = os.path.join(results_folder, '.test_write')
    
    upload_writeable = False
    results_writeable = False
    
    try:
        with open(upload_test_file, 'w') as f:
            f.write('test')
        upload_writeable = True
    except Exception as e:
        current_app.logger.error(f"Cannot write to upload folder: {str(e)}")
    
    try:
        with open(results_test_file, 'w') as f:
            f.write('test')
        results_writeable = True
    except Exception as e:
        current_app.logger.error(f"Cannot write to results folder: {str(e)}")
    
    # Return the information as JSON
    return jsonify({
        'upload_folder': upload_folder,
        'results_folder': results_folder,
        'is_pythonanywhere': is_pythonanywhere,
        'upload_folder_exists': os.path.exists(upload_folder),
        'results_folder_exists': os.path.exists(results_folder),
        'upload_folder_writeable': upload_writeable,
        'results_folder_writeable': results_writeable
    }) 