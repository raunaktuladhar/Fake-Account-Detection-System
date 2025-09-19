// Theme management functions
function initializeTheme() {
    // Get saved theme from localStorage, default to 'light'
    const savedTheme = localStorage.getItem('theme') || 'light';
    
    if (savedTheme === 'dark') {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }
    
    updateThemeIndicator();
}

function toggleTheme() {
    const isDark = document.documentElement.classList.contains('dark');
    const newTheme = isDark ? 'light' : 'dark';
    
    if (newTheme === 'dark') {
        document.documentElement.classList.add('dark');
    } else {
        document.documentElement.classList.remove('dark');
    }
    
    // Save to localStorage
    localStorage.setItem('theme', newTheme);
    
    // Update indicator
    updateThemeIndicator();
    
    // Close dropdown after selection
    hideDropdown();
}

function updateThemeIndicator() {
    const indicator = document.getElementById('theme-indicator');
    if (indicator) {
        const currentTheme = document.documentElement.classList.contains('dark') ? 'Dark' : 'Light';
        indicator.textContent = currentTheme;
    }
}

function hideDropdown() {
    const menu = document.getElementById('user-menu');
    if (menu) {
        menu.classList.add('hidden');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const menu = document.getElementById('user-menu');
    const button = event.target.closest('button');
    
    if (menu && !menu.contains(event.target) && !button?.onclick?.toString().includes('user-menu')) {
        hideDropdown();
    }
});

document.addEventListener("DOMContentLoaded", () => {
    // Initialize theme on page load
    initializeTheme();
    
    const registerForm = document.getElementById("register-form");
    const loginForm = document.getElementById("login-form");
    const profileForm = document.getElementById("profile-form");

    if (registerForm) {
        registerForm.addEventListener("submit", async (event) => {
            event.preventDefault();

            const password = document.getElementById("password").value;
            const confirmPassword = document.getElementById("confirm-password").value;
            const email = document.getElementById("email").value;
            const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;

            if (password !== confirmPassword) {
                alert("Passwords do not match.");
                return;
            }

            if (!emailRegex.test(email)) {
                return;
            }

            console.log("Register form submitted.");
            const formData = new FormData(registerForm);
            
            console.log("Sending registration data to the server...");
            const response = await fetch("/register", {
                method: "POST",
                body: formData,
            });

            console.log("Received response from server:", response);
            const result = await response.json();
            console.log("Response data:", result);

            if (response.ok) {
                console.log("Registration successful. Redirecting to loggedin.html...");
                window.location.href = "/loggedin/loggedin.html";
            } else {
                console.error("Registration failed:", result.error);
                alert("Registration failed: " + result.error);
            }
        });
    }

    if (loginForm) {
        loginForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            const formData = new FormData(loginForm);
            const username = formData.get('username'); // Get username from form
            const response = await fetch("/login", {
                method: "POST",
                body: formData,
            });
            const result = await response.json();
            console.log(result);
            if (response.ok) {
                localStorage.setItem('username', username); // Save username to localStorage
                window.location.href = "/loggedin/loggedin.html";
            } else {
                alert(result.error);
            }
        });
    }

    if (profileForm) {
        profileForm.addEventListener("submit", async function (event) {
            event.preventDefault();

            const formData = new FormData();
            formData.append("screen_name", document.getElementById("screen_name").value.trim());
            formData.append("description", document.getElementById("description").value.trim());
            formData.append("followers_count", document.getElementById("followers_count").value || 0);
            formData.append("friends_count", document.getElementById("friends_count").value || 0);
            formData.append("statuses_count", document.getElementById("statuses_count").value || 0);

            if (!formData.get("screen_name")) {
                alert("Twitter handle is required.");
                return;
            }

            document.getElementById("result-section").style.display = "block";
            document.getElementById("prediction-output").innerText = "Analyzing...";
            document.getElementById("prediction-output").classList.add("loading");

            try {
                const response = await fetch("/predict", {
                method: "POST",
                body: formData,
            });
                const result = await response.json();
                const predictionOutput = document.getElementById("prediction-output");
                predictionOutput.classList.remove("loading", "text-green-500", "text-red-500");

                if (result.prediction === "real") {
                    predictionOutput.classList.add("text-green-500");
                } else {
                    predictionOutput.classList.add("text-red-500");
                }

                predictionOutput.innerText =
                    `Prediction: This profile is ${result.prediction.toUpperCase()} ${
                        result.prediction === "real" ? "✅" : "❌"
                    } (confidence: ${result.confidence[result.prediction].toFixed(2)}%)`;

                const username = localStorage.getItem('username');
                if (username) {
                    const logEntry = {
                        screen_name: formData.get("screen_name"),
                        description: formData.get("description"),
                        followers_count: formData.get("followers_count"),
                        friends_count: formData.get("friends_count"),
                        statuses_count: formData.get("statuses_count"),
                        prediction: result.prediction,
                        confidence: result.confidence[result.prediction]
                    };
                    const logs = JSON.parse(localStorage.getItem(`logSheet_${username}`)) || [];
                    logs.push(logEntry);
                    localStorage.setItem(`logSheet_${username}`, JSON.stringify(logs));
                }

                const existingChart = Chart.getChart("confidence-chart");
                if (existingChart) {
                    existingChart.destroy();
                }
                
                // Update chart colors based on theme
                const isDark = document.documentElement.classList.contains('dark');
                const textColor = isDark ? '#f9fafb' : '#374151';
                const gridColor = isDark ? '#4b5563' : '#e5e7eb';
                
                const ctx = document.getElementById("confidence-chart").getContext("2d");
                new Chart(ctx, {
                    type: "bar",
                    data: {
                        labels: ["Real", "Fake"],
                        datasets: [
                            {
                                label: "Prediction Confidence",
                                data: [
                                    result.confidence.real,
                                    result.confidence.fake
                                ],
                                backgroundColor: ["#16a34a", "#dc2626"],
                                borderColor: ["#14532d", "#991b1b"],
                                borderWidth: 1,
                            },
                        ],
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: { 
                                display: false 
                            },
                            title: { 
                                display: true, 
                                text: "Profile Prediction Confidence",
                                color: textColor
                            },
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                max: 100,
                                title: { 
                                    display: true, 
                                    text: "Confidence (%)",
                                    color: textColor
                                },
                                ticks: {
                                    color: textColor
                                },
                                grid: {
                                    color: gridColor
                                }
                            },
                            x: { 
                                title: { 
                                    display: true, 
                                    text: "Prediction",
                                    color: textColor
                                },
                                ticks: {
                                    color: textColor
                                },
                                grid: {
                                    color: gridColor
                                }
                            },
                        },
                    },
                });
            } catch (error) {
                document.getElementById("prediction-output").classList.remove("loading");
                document.getElementById("prediction-output").classList.add("error");
                document.getElementById("prediction-output").innerText =
                    "Error: Failed to analyze profile.";
                console.error("API error:", error);
            }
        });
    }

    const togglePassword = document.getElementById("toggle-password");
    if (togglePassword) {
        togglePassword.addEventListener("click", function () {
            const passwordInput = document.getElementById("password");
            const eyeOpen = document.getElementById("eye-open");
            const eyeClosed = document.getElementById("eye-closed");
            if (passwordInput.type === "password") {
                passwordInput.type = "text";
                eyeOpen.style.display = "none";
                eyeClosed.style.display = "block";
            } else {
                passwordInput.type = "password";
                eyeOpen.style.display = "block";
                eyeClosed.style.display = "none";
            }
        });
    }

    const toggleConfirmPassword = document.getElementById("toggle-confirm-password");
    if (toggleConfirmPassword) {
        toggleConfirmPassword.addEventListener("click", function () {
            const confirmPasswordInput = document.getElementById("confirm-password");
            const eyeOpenConfirm = document.getElementById("eye-open-confirm");
            const eyeClosedConfirm = document.getElementById("eye-closed-confirm");
            if (confirmPasswordInput.type === "password") {
                confirmPasswordInput.type = "text";
                eyeOpenConfirm.style.display = "none";
                eyeClosedConfirm.style.display = "block";
            } else {
                confirmPasswordInput.type = "password";
                eyeOpenConfirm.style.display = "block";
                eyeClosedConfirm.style.display = "none";
            }
        });
    }

    const passwordField = document.getElementById("password");
    if (passwordField) {
        const strengthBar = document.querySelector(".strength-bar");
        const strengthText = document.querySelector(".strength-text");

        const updatePasswordStrength = () => {
            const password = passwordField.value;
            let strength = 0;
            if (password.length >= 8) strength++;
            if (password.match(/[a-z]/) && password.match(/[A-Z]/)) strength++;
            if (password.match(/[0-9]/)) strength++;
            if (password.match(/[^a-zA-Z0-9]/)) strength++;

            if (strengthBar) {
                strengthBar.style.width = (strength / 4) * 100 + "%";
                switch (strength) {
                    case 0:
                    case 1:
                        strengthBar.style.backgroundColor = "#dc2626";
                        if (strengthText) strengthText.textContent = "Weak";
                        break;
                    case 2:
                        strengthBar.style.backgroundColor = "#f59e0b";
                        if (strengthText) strengthText.textContent = "Medium";
                        break;
                    case 3:
                    case 4:
                        strengthBar.style.backgroundColor = "#16a34a";
                        if (strengthText) strengthText.textContent = "Strong";
                        break;
                }
            }
        };

        passwordField.addEventListener("input", updatePasswordStrength);
    }

    const confirmPasswordField = document.getElementById("confirm-password");
    if (confirmPasswordField) {
        confirmPasswordField.addEventListener("input", function () {
            const password = document.getElementById("password").value;
            const confirmPassword = this.value;
            const errorMessage = document.getElementById("confirm-password-error");
            if (errorMessage) {
                if (password !== confirmPassword) {
                    errorMessage.textContent = "Passwords do not match";
                } else {
                    errorMessage.textContent = "";
                }
            }
        });
    }

    const emailField = document.getElementById("email");
    if (emailField) {
        emailField.addEventListener("input", function () {
            const email = this.value;
            const errorMessage = document.getElementById("email-error");
            const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
            if (errorMessage) {
                if (!emailRegex.test(email)) {
                    errorMessage.textContent = "Invalid email format";
                } else {
                    errorMessage.textContent = "";
                }
            }
        });
    }
});

// Improved Google Sign-In function with better error handling and debugging
async function onSignIn(response) {
    console.log('Google Sign-In callback triggered');
    console.log('Full response object:', response);
    
    try {
        // Extract the credential (JWT token)
        const id_token = response.credential;
        
        if (!id_token) {
            console.error('No credential found in Google response');
            alert('Google Sign-In failed: No token received from Google');
            return;
        }
        
        console.log('Token received (first 50 chars):', id_token.substring(0, 50) + '...');
        
        // Show loading state
        const signInButton = document.querySelector('.g_id_signin');
        if (signInButton) {
            signInButton.style.opacity = '0.5';
            signInButton.style.pointerEvents = 'none';
        }
        
        console.log('Sending token to backend...');
        const res = await fetch("/google-login", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ token: id_token }),
        });
        
        console.log('Response status:', res.status);
        console.log('Response ok:', res.ok);
        
        const result = await res.json();
        console.log('Backend response:', result);
        
        // Reset button state
        if (signInButton) {
            signInButton.style.opacity = '1';
            signInButton.style.pointerEvents = 'auto';
        }
        
        if (res.ok) {
            console.log('Google login successful, redirecting...');
            // Store user info if provided
            if (result.user) {
                localStorage.setItem('username', result.user.name || result.user.email);
                localStorage.setItem('userEmail', result.user.email);
            }
            window.location.href = "/loggedin/loggedin.html";
        } else {
            console.error('Backend error:', result);
            let errorMessage = result.error || 'Unknown error occurred';
            
            // Provide more user-friendly error messages
            if (errorMessage.includes('Invalid Google token')) {
                errorMessage = 'Google authentication failed. Please try again.';
            } else if (errorMessage.includes('Database error')) {
                errorMessage = 'Server error. Please try again later.';
            }
            
            alert('Google Sign-In failed: ' + errorMessage);
        }
    } catch (error) {
        console.error('Error during Google Sign-In:', error);
        
        // Reset button state
        const signInButton = document.querySelector('.g_id_signin');
        if (signInButton) {
            signInButton.style.opacity = '1';
            signInButton.style.pointerEvents = 'auto';
        }
        
        alert('Network error during Google Sign-In. Please check your connection and try again.');
    }
}

// Initialize Google Sign-In when the page loads
window.addEventListener('load', function() {
    console.log('Page loaded, initializing Google Sign-In...');
    
    // Check if Google APIs are loaded
    if (typeof google !== 'undefined' && google.accounts) {
        console.log('Google APIs loaded successfully');
        
        try {
            // Initialize Google Sign-In
            google.accounts.id.initialize({
                client_id: "978383792-ng550h8fkn7aekf2r0uqjn362q0621gp.apps.googleusercontent.com",
                callback: onSignIn,
                auto_select: false,
                cancel_on_tap_outside: true
            });
            
            console.log('Google Sign-In initialized successfully');
            
            // Render the sign-in button if element exists
            const signInDiv = document.querySelector('.g_id_signin');
            if (signInDiv) {
                google.accounts.id.renderButton(signInDiv, {
                    theme: 'outline',
                    size: 'large',
                    type: 'standard',
                    text: 'sign_in_with',
                    shape: 'rectangular',
                    logo_alignment: 'left'
                });
                console.log('Google Sign-In button rendered');
            } else {
                console.warn('Google Sign-In button element not found');
            }
            
            // Optionally show the One Tap prompt (uncomment if you want this)
            // google.accounts.id.prompt();
            
        } catch (error) {
            console.error('Error initializing Google Sign-In:', error);
        }
    } else {
        console.error('Google APIs not loaded. Check if the script is included correctly.');
        // Retry after a short delay
        setTimeout(function() {
            if (typeof google !== 'undefined' && google.accounts) {
                console.log('Google APIs loaded on retry');
                // Repeat initialization here if needed
            } else {
                console.error('Google APIs still not available after retry');
            }
        }, 2000);
    }
});

function confirmLogout() {
    return confirm("Are you sure you want to logout?");
}