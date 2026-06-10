#!/usr/bin/env node
/**
 * Test script to verify SochDB analytics integration (Node.js).
 * 
 * This sends several test events to PostHog to verify the integration works.
 * Events should appear in PostHog dashboard at: https://us.posthog.com
 */

// Ensure analytics is enabled
process.env.SOCHDB_DISABLE_ANALYTICS = 'false';

const {
  captureAnalytics,
  captureError,
  trackDatabaseOpen,
  trackVectorSearch,
  trackBatchInsert,
  isAnalyticsDisabled
} = require('./sochdb-js/dist/cjs/index.js');

async function main() {
  console.log('='.repeat(60));
  console.log('SochDB Analytics Test (Node.js)');
  console.log('='.repeat(60));
  console.log();
  
  // Check status
  console.log(`Analytics disabled: ${isAnalyticsDisabled()}`);
  console.log();
  
  console.log('Sending test events...');
  console.log();
  
  try {
    // Test 1: Basic event
    console.log('1. Basic event...');
    await captureAnalytics('analytics_test_js', {
      test_number: 1,
      test_name: 'basic_event_js',
      timestamp: Date.now()
    });
    console.log('   ✅ Sent');
    
    // Test 2: Database open
    console.log('2. Database open event...');
    await trackDatabaseOpen('./test_db', 'embedded');
    console.log('   ✅ Sent');
    
    // Test 3: Vector search
    console.log('3. Vector search event...');
    await trackVectorSearch(1536, 10, 45.7);
    console.log('   ✅ Sent');
    
    // Test 4: Batch insert
    console.log('4. Batch insert event...');
    await trackBatchInsert(1000, 768, 123.4);
    console.log('   ✅ Sent');
    
    // Test 5: Error event
    console.log('5. Error event...');
    await captureError('test_error_js', 'This is a test error for analytics verification');
    console.log('   ✅ Sent');
    
    console.log();
    console.log('='.repeat(60));
    console.log('All test events sent successfully!');
    console.log();
    console.log('Check PostHog dashboard:');
    console.log('https://us.posthog.com/project/YOUR_PROJECT_ID/events');
    console.log();
    console.log('Expected events:');
    console.log('  - analytics_test_js');
    console.log('  - database_opened');
    console.log('  - vector_search');
    console.log('  - batch_insert');
    console.log('  - error');
    console.log('='.repeat(60));
  } catch (err) {
    console.error('Error:', err);
    process.exit(1);
  }
}

main();
