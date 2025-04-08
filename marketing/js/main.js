// AMTA Marketing Website JavaScript

document.addEventListener('DOMContentLoaded', () => {
    // Smooth scrolling for anchor links
    const anchorLinks = document.querySelectorAll('a[href^="#"]');
    
    anchorLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 80, // Adjust for header
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Animate timeline items on scroll
    const timelineItems = document.querySelectorAll('.timeline-item');
    
    const animateOnScroll = () => {
        timelineItems.forEach(item => {
            const itemPosition = item.getBoundingClientRect().top;
            const screenPosition = window.innerHeight / 1.2;
            
            if (itemPosition < screenPosition) {
                item.classList.add('animate');
            }
        });
    };
    
    // Add animation class to timeline items
    timelineItems.forEach(item => {
        item.style.opacity = '0';
        item.style.transform = 'translateY(20px)';
        item.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    });
    
    // Animate on scroll
    window.addEventListener('scroll', animateOnScroll);
    
    // Initial check for animations on page load
    animateOnScroll();
}); 