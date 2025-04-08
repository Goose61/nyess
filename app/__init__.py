import os
from flask import Flask, render_template
from flask_cors import CORS
from flask_login import LoginManager

def create_app(test_config=None):
    """Create and configure the Flask application."""
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    # Enable CORS
    CORS(app)
    
    # Determine if running on PythonAnywhere
    is_pythonanywhere = 'PYTHONANYWHERE_DOMAIN' in os.environ
    
    # Set default configuration
    if is_pythonanywhere:
        # Use PythonAnywhere's file structure
        # /home/yourusername is the home directory on PythonAnywhere
        username = os.path.basename(os.path.expanduser('~'))
        app.config.from_mapping(
            SECRET_KEY='dev',
            UPLOAD_FOLDER=os.path.join('/home', username, 'ifc_uploads'),
            RESULTS_FOLDER=os.path.join('/home', username, 'ifc_results'),
            ALLOWED_EXTENSIONS={'ifc'},
            MAX_CONTENT_LENGTH=100 * 1024 * 1024,  # 100MB max upload
            IS_PYTHONANYWHERE=True
        )
        # Ensure the folders exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
        # Log the paths for debugging
        print(f"Running on PythonAnywhere with upload folder: {app.config['UPLOAD_FOLDER']}")
        print(f"Results folder: {app.config['RESULTS_FOLDER']}")
    else:
        # Standard configuration for local development
        app.config.from_mapping(
            SECRET_KEY='dev',
            UPLOAD_FOLDER=os.path.join(os.getcwd(), 'app', 'uploads'),
            RESULTS_FOLDER=os.path.join(os.getcwd(), 'app', 'results'),
            ALLOWED_EXTENSIONS={'ifc'},
            MAX_CONTENT_LENGTH=100 * 1024 * 1024,  # 100MB max upload
            IS_PYTHONANYWHERE=False
        )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    
    # Initialize the login manager
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'warning'
    login_manager.init_app(app)
    
    # Import and initialize User model
    from app.models.user import User
    User.init_db()
    
    @login_manager.user_loader
    def load_user(user_id):
        # Load the user from the database
        return User.get_by_id(int(user_id))

    # Register blueprints
    from app.routes import main, api, errors
    app.register_blueprint(main.bp)
    app.register_blueprint(api.bp)
    app.register_blueprint(errors.bp)
    
    # Register auth blueprint
    from app.routes import auth
    app.register_blueprint(auth.bp)

    # Register error handlers
    from app.routes.errors import not_found_error, internal_error, forbidden_error, too_large_error, bad_request_error
    app.register_error_handler(404, not_found_error)
    app.register_error_handler(500, internal_error)
    app.register_error_handler(403, forbidden_error)
    app.register_error_handler(413, too_large_error)
    app.register_error_handler(400, bad_request_error)

    # Add a health check route
    @app.route('/health')
    def health_check():
        return {'status': 'ok'}, 200
        

    return app 