import os
from flask import (
    Blueprint, flash, redirect, render_template, request, 
    url_for, current_app, session
)
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required, current_user
from app.models.user import User

bp = Blueprint('auth', __name__, url_prefix='/auth')

@bp.route('/register', methods=('GET', 'POST'))
def register():
    """Register a new user."""
    # If already logged in, redirect to index
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        error = None
        
        if not username:
            error = 'Username is required.'
        elif not email:
            error = 'Email is required.'
        elif not password:
            error = 'Password is required.'
        elif password != confirm_password:
            error = 'Passwords do not match.'
        elif User.get_by_username(username):
            error = f"User {username} is already registered."
        elif User.get_by_email(email):
            error = f"Email {email} is already registered."
            
        if error is None:
            # Create new user
            user = User(username=username, email=email)
            user.set_password(password)
            user.save()
            
            # Log in the new user
            login_user(user)
            
            # Redirect to index
            return redirect(url_for('main.index'))
            
        flash(error, 'danger')
            
    return render_template('auth/register.html')
    
@bp.route('/login', methods=('GET', 'POST'))
def login():
    """Log in a user."""
    # If already logged in, redirect to index
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
        
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        remember = 'remember' in request.form
        
        error = None
        user = User.get_by_username(username)
        
        if user is None:
            error = 'Incorrect username.'
        elif not user.check_password(password):
            error = 'Incorrect password.'
            
        if error is None:
            # Log in the user
            login_user(user, remember=remember)
            
            # Get the next page from the query string
            next_page = request.args.get('next')
            
            # Validate the next page (prevent open redirects)
            if not next_page or not next_page.startswith('/'):
                next_page = url_for('main.index')
                
            # Redirect to the next page
            return redirect(next_page)
            
        flash(error, 'danger')
            
    return render_template('auth/login.html')
    
@bp.route('/logout')
@login_required
def logout():
    """Log out a user."""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.index')) 