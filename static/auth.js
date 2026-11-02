(() => {
  const $ = (id) => document.getElementById(id);
  const loginForm = $('loginForm'), registerForm = $('registerForm'), verifyForm = $('verifyForm');
  const notice = $('notice'), signedIn = $('signedIn');
  let currentMode = 'login';
  function showNotice(message, type = 'success') {
    notice.textContent = message; notice.className = `notice ${type}`;
  }
  function hideNotice() { notice.className = 'notice hidden'; notice.textContent = ''; }
  function mode(name) {
    currentMode = name; hideNotice(); signedIn.classList.add('hidden');
    loginForm.classList.toggle('hidden', name !== 'login');
    registerForm.classList.toggle('hidden', name !== 'register');
    verifyForm.classList.toggle('hidden', name !== 'verify');
    $('loginTab').classList.toggle('active', name === 'login');
    $('registerTab').classList.toggle('active', name === 'register');
    $('loginTab').classList.toggle('hidden', name === 'verify');
    $('registerTab').classList.toggle('hidden', name === 'verify');
    $('title').textContent = name === 'register' ? 'Create your account' : name === 'verify' ? 'Check your email' : 'Welcome back';
    $('intro').textContent = name === 'register' ? 'Set up your secure Stock AI account.' : name === 'verify' ? 'Enter the six-digit code we sent you.' : 'Sign in to continue to your dashboard.';
  }
  async function post(path, body) {
    const response = await fetch(path, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || 'Something went wrong. Please try again.');
    return payload;
  }
  $('loginTab').addEventListener('click', () => mode('login'));
  $('registerTab').addEventListener('click', () => mode('register'));
  $('backToRegister').addEventListener('click', () => mode('register'));
  registerForm.addEventListener('submit', async (event) => {
    event.preventDefault(); hideNotice();
    const body = Object.fromEntries(new FormData(registerForm));
    try {
      const result = await post('/api/register', body);
      verifyForm.elements.email.value = body.email;
      mode('verify'); showNotice(result.message || 'Verification code sent to your email.');
    } catch (error) { showNotice(error.message, 'error'); }
  });
  verifyForm.addEventListener('submit', async (event) => {
    event.preventDefault(); hideNotice();
    const body = Object.fromEntries(new FormData(verifyForm));
    try {
      const result = await post('/api/verify-email', body);
      mode('login'); loginForm.elements.email.value = body.email;
      showNotice(result.message || 'Email verified. You can now sign in.');
    } catch (error) { showNotice(error.message, 'error'); }
  });
  loginForm.addEventListener('submit', async (event) => {
    event.preventDefault(); hideNotice();
    const body = Object.fromEntries(new FormData(loginForm));
    try {
      const result = await post('/api/login', body);
      sessionStorage.setItem('stock_ai_token', result.token);
      signedIn.classList.remove('hidden');
      loginForm.classList.add('hidden'); registerForm.classList.add('hidden'); verifyForm.classList.add('hidden');
      $('loginTab').classList.add('hidden'); $('registerTab').classList.add('hidden');
      $('title').textContent = 'You’re all set'; $('intro').textContent = 'Your Stock AI session is active.';
      $('signedEmail').textContent = result.user.email; hideNotice();
    } catch (error) { showNotice(error.message, 'error'); }
  });
  $('logout').addEventListener('click', () => {
    sessionStorage.removeItem('stock_ai_token'); signedIn.classList.add('hidden'); mode('login'); showNotice('You have signed out.');
  });
})();
