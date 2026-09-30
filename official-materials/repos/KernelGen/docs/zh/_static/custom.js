// _static/custom.js
document.addEventListener('DOMContentLoaded', function() {
    // 直接替换按钮文本
    const repoTexts = document.querySelectorAll('.btn-source-repository-button .btn__text-container');
    const editTexts = document.querySelectorAll('.btn-source-edit-button .btn__text-container');
    
    repoTexts.forEach(el => el.textContent = '代码仓库');
    editTexts.forEach(el => el.textContent = '建议编辑');
    
    // 替换tooltip文本
    document.querySelectorAll('[data-bs-original-title="源码库"]')
        .forEach(el => el.setAttribute('data-bs-original-title', '源码仓库'));
    
    document.querySelectorAll('[data-bs-original-title="Suggest edit"]')
        .forEach(el => el.setAttribute('data-bs-original-title', '建议编辑'));
});


// _static/custom.js
document.addEventListener('DOMContentLoaded', function() {
    // 1. 替换 Repository 和 Suggest edit 按钮文本
    const repoTexts = document.querySelectorAll('.btn-source-repository-button .btn__text-container');
    const editTexts = document.querySelectorAll('.btn-source-edit-button .btn__text-container');
    
    repoTexts.forEach(el => el.textContent = '代码仓库');
    editTexts.forEach(el => el.textContent = '建议编辑');
    
    // 2. 替换 tooltip 文本
    document.querySelectorAll('[data-bs-original-title="源码库"]')
        .forEach(el => el.setAttribute('data-bs-original-title', '源码仓库'));
    
    document.querySelectorAll('[data-bs-original-title="Suggest edit"]')
        .forEach(el => el.setAttribute('data-bs-original-title', '建议编辑'));
    
    // 3. 汉化 Light/Dark 主题切换按钮
    const themeButtons = document.querySelectorAll('.theme-switch-button');
    
    themeButtons.forEach(button => {
        // 替换 tooltip 文本
        if (button.getAttribute('data-bs-original-title') === 'light/dark') {
            button.setAttribute('data-bs-original-title', '日间/黑夜');
        }
        
        // 替换 aria-label 属性
        if (button.getAttribute('aria-label') === 'light/dark') {
            button.setAttribute('aria-label', '日间/黑夜');
        }
        
        // 如果有 title 属性也替换
        if (button.getAttribute('title') === 'light/dark') {
            button.setAttribute('title', '日间/黑夜');
        }
    });
});