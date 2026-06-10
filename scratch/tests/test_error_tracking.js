// Test error tracking in JavaScript/Node.js

const { captureError, isAnalyticsDisabled } = require('./sochdb-js/dist/cjs/index.js');

async function testErrorTracking() {
  console.log('='.repeat(60));
  console.log('Testing Error Tracking (Static Only)');
  console.log('='.repeat(60));
  console.log(`Analytics disabled: ${isAnalyticsDisabled()}\n`);

  if (!isAnalyticsDisabled()) {
    console.log('Sending test error events (static info only)...\n');

    const errors = [
      ['connection_error', 'database.open'],
      ['query_error', 'sql.execute'],
      ['permission_error', 'table.access'],
      ['timeout_error', 'transaction.commit'],
    ];

    for (let i = 0; i < errors.length; i++) {
      const [errorType, location] = errors[i];
      console.log(`${i + 1}. ${errorType} @ ${location}...`);
      await captureError(errorType, location);
      console.log(`   ✅ Sent`);
    }

    console.log('\n' + '='.repeat(60));
    console.log('All error events sent successfully!');
    console.log('Only static info sent - no sensitive data!');
    console.log('='.repeat(60));
  } else {
    console.log('⚠️  Analytics disabled - no events sent');
  }
}

testErrorTracking().catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
