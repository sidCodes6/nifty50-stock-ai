(() => {
  const $ = (id) => document.getElementById(id);
  const loginForm = $('loginForm'), registerForm = $('registerForm'), notice = $('notice');
  function showNotice(message, type = 'success') { notice.textContent = message; notice.className = `notice ${type}`; }
  function mode(name) {
    notice.className = 'notice hidden';
    loginForm.classList.toggle('hidden', name !== 'login');
    registerForm.classList.toggle('hidden', name !== 'register');
    $('loginTab').classList.toggle('active', name === 'login');
    $('registerTab').classList.toggle('active', name === 'register');
    $('title').textContent = name === 'register' ? 'Create your account' : 'Welcome back';
    $('intro').textContent = name === 'register' ? 'Create your Stock AI account.' : 'Sign in to continue to your dashboard.';
  }
  async function handoff(user) {
    const idToken = await user.getIdToken(true);
    const response = await fetch('/api/firebase-login', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({idToken})});
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.error || 'Could not verify Firebase sign-in.');
    window.location.assign(`${result.dashboard_url}/?ticket=${encodeURIComponent(result.dashboard_ticket)}`);
  }
  $('loginTab').addEventListener('click', () => mode('login'));
  $('registerTab').addEventListener('click', () => mode('register'));
  $('google-signin-btn').addEventListener('click', async () => {
    try { const provider = new firebase.auth.GoogleAuthProvider(); const result = await firebase.auth().signInWithPopup(provider); await handoff(result.user); }
    catch (error) { showNotice(error.message || 'Google sign-in failed.', 'error'); }
  });
  loginForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    try { const user = await firebase.auth().signInWithEmailAndPassword(loginForm.elements.email.value, loginForm.elements.password.value); await handoff(user.user); }
    catch (error) { showNotice(error.message || 'Sign-in failed.', 'error'); }
  });
  registerForm.addEventListener('submit', async (event) => {
    event.preventDefault();
    try {
      const result = await firebase.auth().createUserWithEmailAndPassword(registerForm.elements.email.value, registerForm.elements.password.value);
      await handoff(result.user);
    } catch (error) { showNotice(error.message || 'Account creation failed.', 'error'); }
  });
})();
