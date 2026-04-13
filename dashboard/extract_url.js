import https from 'https';

fetch('https://app.spline.design/file/9df9601f-9999-4c76-b44f-8328b500f7b8')
  .then(res => res.text())
  .then(html => {
    const regex = /https:\/\/[a-z0-9-]+\.spline\.design\/[^"']+/g;
    const matches = html.match(regex);
    if (matches) {
      console.log(Array.from(new Set(matches)));
    } else {
      console.log("No matches found.");
    }
  });